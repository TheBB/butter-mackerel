from datetime import datetime, timedelta
import functools
import inflect
import numpy as np
from os import path
import pickle
import random
from scipy.stats import binom

from butter.db import rsync_file
from butter.programs import Slideshow, FromPicker, bind
import butter.plugin as plugin


def sub_value(secs, threshold, limit, spread):
    secs = (secs - threshold) / spread
    h = lambda x: np.exp(-1/x) if x > 0 else 0
    g = h(secs) / (h(secs) + h(1-secs))
    g *= np.arctan(secs) / np.pi + 1/2
    return float(g * limit)


def limiting_sub(current, limit, subtract, cost=1, effectiveness=1):
    available = effectiveness * subtract
    while current > limit and available > cost:
        available -= cost
        current -= 1
    return current, available // effectiveness


class Timed:

    def __init__(self, func, argname):
        self.func = func
        self.argname = argname
        self.last_call = None

    def __call__(self, *args, **kwargs):
        now = datetime.now()
        if self.last_call is None:
            retval = self.func(*args, **kwargs, **{self.argname: None})
        else:
            duration = now - self.last_call
            retval = self.func(*args, **kwargs, **{self.argname: duration})
        self.last_call = now
        return retval


def timed(argname):
    def decorator(func):
        return Timed(func, argname)
    return decorator


class MackerelSlideshow(Slideshow):

    def __init__(self, state, picker, m):
        self.state = state
        state.m = m
        self.initialized = False
        super().__init__(m, picker)
        self.update_msg()
        self.initialized = True
        self.state.pop(m)
        self.last_tick = datetime.now()

    def make_current(self, m):
        self.update_msg()

    def update_msg(self, picid=None):
        s = self.state
        p = inflect.engine()
        msg = f'{s.permissions} {p.plural("permission", s.permissions)}, score: {s.score}, coins: {s.coins}'
        if s.nqueued:
            msg += f' | {s.nqueued} {p.plural("event", s.nqueued)} queued'
        if picid is not None:
            msg += f' ({picid:08})'

        self.message = msg

    @bind()
    def pic(self, m):
        pic = super().pic(m, set_msg=False)
        cfg = self.state.cfg['tick']

        live = not m.safe and m.height() > cfg['min-height'] and self.initialized
        now = datetime.now()
        dt = None
        if live and self.last_tick:
            now = datetime.now()
            dt = now - self.last_tick
            self.last_tick = now

            if not (timedelta(seconds=cfg['min-seconds']) <= dt <= timedelta(seconds=cfg['max-seconds'])):
                dt = None
        elif live:
            self.last_tick = now
        else:
            self.last_tick = None

        self.state.tick(dt)
        if self.initialized:
            self.state.pop(m)
        self.update_msg(pic.id)

        if random.random() < self.state.cfg['flash']['prob'] and pic.is_still:
            m.flash(random.choice(self.state.cfg['flash']['messages']))

    @bind('m')
    def mas(self, m):
        self.state.mas(m, physical=False)
        self.update_msg()

    @bind('M')
    def mas_physical(self, m):
        self.state.mas(m, physical=True)
        self.update_msg()

    @bind('G')
    def thing(self, m):
        BestOfGame(self.state, m)

    @bind('B')
    def shop(self, m):
        items = {key: val for key, val in zip('abcdefghijklmnopqrstuvwxyz', self.state.cfg['shop'])}
        text = [f"{key.upper()}: {val['text']} ({val['price']})" for key, val in items.items()]
        retval = self.m.popup_message(text).lower()
        if retval not in items:
            return
        if retval in items:
            self.state.buy(items[retval])
        self.update_msg()


del MackerelSlideshow.keymap['P']
del MackerelSlideshow.keymap['c']


class BestOfGame(FromPicker):

    def __init__(self, state, m):
        cfg = state.cfg['bestof']
        super().__init__(m, m.db.picker(cfg['picker']))

        self.state = state
        self.trigger = lambda pic: pic.eval(cfg['trigger'])
        self.multiplier = lambda pic: pic.eval(cfg['multiplier'])
        self.prob_reject = cfg['rejection_prob']
        self.cfg = cfg

        self.count = 0

        self.max_pts = self.cfg['maxpts']
        self.pts = state.game_state
        # self.prev_winner = None
        # self.speed, self.bias = 0.55, 0.0

        self.update_msg()

    @property
    def current(self):
        return self.state.game_current

    @current.setter
    def current(self, value):
        self.state.game_current = value

    def update_msg(self, picid=None):
        player = ['we', 'you'][self.current]
        msg = (
            '  ·  '.join(f'{we}–{you}' for we, you in zip(*self.pts)) +
            '  ·  ' + str(self.state.game_count) + '/' + str(self.cfg['cycles']) +
            f'     ({self.state.score})' +
            f' ({self.count} · {player})'
        )
        if picid is not None:
            msg += f' ({picid:08})'
        self.message = msg

    def add_pts(self, winner, npts):
        l_pts, w_pts = self.pts[1 - winner], self.pts[winner]
        for _ in range(npts):
            for i in range(len(self.max_pts)):
                if i > 0:
                    w_pts[i-1] = l_pts[i-1] = 0
                w_pts[i] += 1
                if w_pts[i] != self.max_pts[i]:
                    break
            if w_pts[-1] == self.max_pts[-1]:
                self.state.add_score(-1 if winner == 1 else 1, msg=False)
                self.state.complete_game()
                w_pts[:] = [0] * len(w_pts)
                l_pts[:] = [0] * len(l_pts)

    @bind('G')
    def exit(self, m):
        m.pop(self)

    @bind()
    @timed('dt')
    def pic(self, m, dt):
        if dt and dt.total_seconds() < self.cfg['time']:
            return

        self.current = 1 - self.current
        cur = self.current

        pic = self.picker.get()
        while not self.trigger(pic) and random.random() <= self.prob_reject:
            pic = self.picker.get()

        self.count += 1
        super().pic(m, set_msg=False, pic=pic)

        if self.trigger(pic):
            npts = self.count * self.multiplier(pic)
            self.add_pts(cur, npts)
            self.update_msg(pic.id)
            e = inflect.engine()
            msg = ['{} {} for {}'.format(npts, e.plural('point', npts), 'us' if cur == 0 else 'you')]
            m.popup_message(msg)
            self.count = 0
            self.current = 1 - self.current

        else:
            self.update_msg(pic.id)


def visible(func):
    func.visible = True
    return func


class MackerelState:

    def __init__(self, loader):
        self.cfg = loader.cfg['mackerel']
        local = path.join(loader.path, 'mackerel.state')
        self._live = False

        if loader.remote:
            remote = path.join(loader.remote, 'mackerel.state')
            rsync_file(remote, local)

        with open(local, 'rb') as f:
            self._state = pickle.load(f)

        self._loader = loader
        globs = {key: getattr(self, key) for key in dir(self)}
        self._globals = {
            key: val for key, val in globs.items()
            if hasattr(val, 'visible') and val.visible
        }
        self._globals.update({
            'binomial': np.random.binomial,
            'logistic': np.random.logistic,
            'normal': np.random.normal,
            'poisson': np.random.poisson,
            'random': np.random.random,
            'uniform': np.random.uniform,
        })

    def __del__(self):
        print(self._state)
        local = path.join(self._loader.path, 'mackerel.state')
        with open(local, 'wb') as f:
            pickle.dump(self._state, f)
        if self._loader.remote:
            remote = path.join(self._loader.remote, 'mackerel.state')
            rsync_file(local, remote)

    def update(self):
        if hasattr(self, 'm'):
            self.m.blur = self._state['blur']

    def execute(self, code, globs=None):
        globs = dict(globs or {})
        globs.update(self._globals)
        exec(code, globs, self._state)
        self.update()

    def evaluate(self, code, globs=None):
        globs = dict(globs or {})
        globs.update(self._globals)
        retval = eval(code, globs, self._state)
        self.update()
        return retval

    def tick(self, live_duration=None):
        now = datetime.now()
        dt_long = (now - self._state['last_tick']).total_seconds()
        dt_short = dt_long if self._live else 0.0
        self._live = True

        self._state['last_tick'] = now

        scale_long = self.cfg['tick']['rate'] * dt_long / 24 / 60 / 60
        scale_short = self.cfg['tick']['rate'] * dt_short / 24 / 60 / 60

        events = []
        for rate, event in self.cfg['tick']['events']:
            events += [event] * np.random.poisson(rate * scale_long)
        for rate, event in self.cfg['tick']['live-events']:
            events += [event] * np.random.poisson(rate * scale_short)
        random.shuffle(events)
        self._state['queue'].extend(events)

    def pop(self, m):
        self.update()
        if not self._state['queue']:
            return
        event = self._state['queue'].pop(0)
        m.blur = self._state['blur']
        self.execute(event)

    def event(self, name, m, **kwargs):
        choices = self.cfg['events'][name]
        weights, population = zip(*choices)
        event, = random.choices(population, weights=weights, k=1)
        print('Event:', name, event, kwargs)
        self.execute(event, kwargs)

    def complete_game(self):
        self._state['game_count'] += 1
        if self._state['game_count'] == self.cfg['bestof']['cycles']:
            self.add_score(-1, msg=False)
            self._state['game_count'] = 0

    @property
    def game_count(self):
        return self._state['game_count']

    @property
    def game_state(self):
        return self._state['game_state']

    @property
    def game_current(self):
        return self._state['game_current']

    @game_current.setter
    def game_current(self, value):
        self._state['game_current'] = value

    def game_done(self):
        return self._state['game_count'] == 0

    @property
    def permissions(self):
        return self._state['permissions']

    @property
    def score(self):
        return self._state['score']

    @property
    def nqueued(self):
        return len(self._state['queue'])

    @property
    def coins(self):
        return self._state['coins']

    def mas(self, m, physical=False):
        self.add_permissions(-1, msg=False)
        if physical or self.permissions >= 0:
            m.popup_message('Removed one permission')
        else:
            self.event('illegal-mas', m)

    @visible
    def add_permissions(self, new=1, msg=True):
        self._state['permissions'] = min(self.permissions + new, 1)
        if msg:
            self.m.popup_message(f'Added permission: {new}, now {self.permissions}')
        return True

    @visible
    def add_score(self, new, msg=True):
        self._state['score'] = max(self._state['score'] + new, 0)
        if msg:
            self.m.popup_message(f'Added score: {new}, now {self.score}')

    @visible
    def add_coins(self, new, msg=True):
        self._state['coins'] += new

    def buy(self, item):
        if self.coins < item['price']:
            return
        self._state['coins'] = self.coins - item['price']
        self.execute(item['event'])

    @visible
    def nothing(self, msg=True):
        if msg:
            self.m.popup_message('Nothing')


class Mackerel(plugin.PluginBase):

    def __init__(self):
        super().__init__()

    def activate(self):
        super().activate()
        self.state = MackerelState(self.loader)
        self.cfg = self.loader.cfg['mackerel']

    @property
    def picker(self):
        with self.loader.database() as db:
            return db.picker(self.cfg['picker'])

    def deactivate(self):
        super().deactivate()
        del self.state

    def get_default_program(self):
        return functools.partial(MackerelSlideshow, self.state, self.picker)

    def add_failed(self, filename, reason):
        if reason == 'collision':
            self.state.event('add-collision', None, msg=False)

    def add_succeeded(self, pic):
        self.state.add_coins(pic.eval(self.cfg['add']), msg=False)
