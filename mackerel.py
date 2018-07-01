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
        self.last_tick = datetime.now()

    def make_current(self, m):
        self.update_msg()

    def update_msg(self):
        s = self.state
        p = inflect.engine()
        msg = f'{s.permissions} {p.plural("permission", s.permissions)}, score: {s.score}, coins: {s.coins}'
        if s.nqueued:
            msg += f' | {s.nqueued} {p.plural("event", s.nqueued)} queued'

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
        self.update_msg()

    @bind('m')
    def mas(self, m):
        self.state.mas(m, physical=False)
        self.update_msg()

    @bind('M')
    def mas_physical(self, m):
        self.state.mas(m, physical=True)
        self.update_msg()

    def _game_callback(self, m, winner, npts):
        self.state.add_score(-20, msg=False, actual=False)
        if winner == 0:
            self.state.game_against(m, npts)
        else:
            self.state.game_for(m, npts)

    @bind('G')
    def thing(self, m):
        self.state.add_score(20, msg=False, actual=False)
        BestOfGame(self.state, m, functools.partial(self._game_callback, m))

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

    def __init__(self, state, m, callback, start=(0,0)):
        self.callback = callback
        cfg = state.cfg['bestof']
        super().__init__(m, m.db.picker(cfg['picker']))

        self.trigger = lambda pic: pic.eval(cfg['trigger'])
        self.value = lambda pic: pic.eval(cfg['value'])
        self.cfg = cfg

        self.max_pts = self.cfg['maxpts']
        self.pts = [[0] * len(self.max_pts) for _ in range(2)]
        self.pts[0][-1] = start[0]
        self.pts[1][-1] = start[1]
        self.current = random.choice([0,1])
        self.prev_winner = None
        self.speed, self.bias, self.add_bias = 0.55, 0.0, 0.0
        self.done = False

        self.update_msg()

    def update_msg(self):
        self.message = (
            '  ·  '.join(f'{we}–{you}' for we, you in zip(*self.pts)) +
            f'     ({self.add_bias:.2f})'
        )

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
                break

    def win(self, m, winner, npts):
        e = inflect.engine()
        msg = '{} win with {} {}'.format('We' if winner == 0 else 'You', npts, e.plural('point', npts))
        m.popup_message(msg)
        self.callback(winner, npts)
        self.done = True
        m.pop(self)

    @bind()
    @timed('dt')
    def pic(self, m, dt):
        cur = self.current
        p = lambda b: max(min((1.020**b) / (1.020**b + 1), 0.93), 0.07)
        conv = lambda p: self.speed * p

        if dt and dt.total_seconds() < self.cfg['time']:
            return

        self.update_msg()
        bias = self.bias + self.add_bias
        prob_win = conv(p(bias) if cur == 0 else 1 - p(bias))
        win = random.random() <= prob_win
        pic = self.picker.get()
        while self.trigger(pic) != win:
            pic = self.picker.get()

        pic = super().pic(m, set_msg=False, pic=pic)

        if not win:
            self.current = 1 - cur
            return

        npts = self.value(pic)
        self.add_pts(cur, npts)
        self.update_msg()

        if self.pts[cur][-1] == self.max_pts[-1]:
            self.win(m, cur, self.pts[cur][-1] - self.pts[1-cur][-1])
            return

        sign = 1 if cur == 0 else -1
        if self.prev_winner != cur:
            self.bias = 0.0
        else:
            self.bias += sign * min(npts, self.cfg['threshold'])

        e = inflect.engine()
        msg = ['{} {} for {}'.format(npts, e.plural('point', npts), 'us' if cur == 0 else 'you')]

        p_t, p_f = conv(p(self.bias)), conv(1 - p(self.bias))
        denom = p_t + p_f - p_t * p_f
        if cur == 0:
            p_t /= denom
        else:
            p_t = 1 - p_f / denom
        p_f = 1 - p_t
        msg.append(f'{100*p_t:.2f}% – {100*(1-p_t):.2f}%')

        self.prev_winner = cur
        self.update_msg()
        m.popup_message(msg)


def visible(func):
    func.visible = True
    return func


class MackerelState:

    def __init__(self, loader):
        self.cfg = loader.cfg['mackerel']
        local = path.join(loader.path, 'mackerel.state')

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

    def execute(self, code, globs=None):
        globs = dict(globs or {})
        globs.update(self._globals)
        exec(code, globs, self._state)

    def evaluate(self, code, globs=None):
        globs = dict(globs or {})
        globs.update(self._globals)
        return eval(code, globs, self._state)

    def tick(self, live_duration=None):
        now = datetime.now()
        dt = (now - self._state['last_tick']).total_seconds()
        self._state['last_tick'] = now

        scale = self.cfg['tick']['rate'] / sum(freq for freq, _ in self.cfg['tick']['events'])
        scale *= dt / 24 / 60 / 60

        events = []
        for rate, event in self.cfg['tick']['events']:
            events += [event] * np.random.poisson(rate * scale)
        random.shuffle(events)
        self._state['queue'].extend(events)

    def pop(self, m):
        if not self._state['queue']:
            return
        event = self._state['queue'].pop(0)
        self.m = m
        self.execute(event)

    def event(self, name, m, **kwargs):
        choices = self.cfg['events'][name]
        weights, population = zip(*choices)
        event, = random.choices(population, weights=weights, k=1)
        print('Event:', name, event, kwargs)
        self.execute(event, kwargs)

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

    def game_for(self, m, npts):
        self.event('game-for', m, npts=npts)

    def game_against(self, m, npts):
        self.event('game-against', m, npts=npts)

    @visible
    def add_permissions(self, new=1, msg=True):
        self._state['permissions'] = min(self.permissions + new, 1)
        if msg:
            self.m.popup_message(f'Added permission: {new}, now {self.permissions}')

    @visible
    def add_score(self, new, msg=True, actual=True):
        if self._state['score'] == 0 and new < 0:
            self.add_permissions(msg=msg)
            self._state['score'] = self.cfg['score']['base']
            return

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
