import click
from collections import namedtuple
from datetime import datetime, timedelta
import functools
import inflect
import numpy as np
from os import path
import pickle
import random
from scipy.stats import binom
import string
import yaml

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


class MackerelSlideshow(Slideshow):

    def __init__(self, state, picker, m):
        self.state = state
        state.m = m
        self.initialized = False
        super().__init__(m, picker)
        self.update_msg()
        self.initialized = True

    def make_current(self, m):
        self.update_msg()

    @property
    def locked(self):
        return self.lock < self.lock_limit

    def update_msg(self):
        s = self.state
        p = inflect.engine()
        msg = f'{s.permissions} {p.plural("permission", s.permissions)}'
        if s.nqueued:
            msg += f' | {s.nqueued} {p.plural("event", s.nqueued)} queued'
        if s.waiting:
            msg += f' | waiting {s.waiting} ({"closed" if s.closed else "open"})'

        self.message = msg

    @bind()
    def pic(self, m):
        pic = super().pic(m, set_msg=False)
        self.state.tick()
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
        if winner == 0:
            self.state.game_against(m, npts)
        else:
            self.state.game_for(m, npts)

    @bind('g')
    def thing(self, m):
        if self.state.waiting:
            m.popup_message("Nope")
        else:
            BestOfGame(self.state, m, functools.partial(self._game_callback, m))

    def _bet_callback(self, m, your_reduction, our_reduction, winner, npts):
        if winner == 0:
            self.state.add_wait(our_reduction, msg=False)
        else:
            self.state.add_wait(-your_reduction, msg=False)
        if self.state.waiting:
            self.state.close_wait(msg=False)

    @bind('b')
    def bet(self, m):
        max_duration = self.state.waiting
        if not max_duration:
            m.popup_message("Nope")
            return
        max_duration /= 2
        your_reduction = max_duration
        bias = 0

        maxpts = self.state.cfg['bestof']['maxpts'][-1]

        while True:
            a = maxpts - max(bias, 0)
            b = maxpts + min(bias, 0)
            nrounds = a + b - 1
            prob = sum(binom.pmf(n, nrounds, 0.5) for n in range(a, nrounds+1))
            our_reduction = (1 - prob) * your_reduction * 1.1 / prob

            ret = m.popup_message([
                f'Maximal allowable reduction: {max_duration}',
                f'Requested reduction: {your_reduction}',
                f'Start score: {max(bias,0)}-{-min(bias,0)}',
                f'We win in {prob*100:.2f}% of cases',
                f'Our reduction: {our_reduction}'
            ])
            if ret == 'RET':
                break
            elif ret in {'ESC', 'q'}:
                return
            elif ret in 'mMhHdD':
                if ret.lower() == 'm':
                    adjust = timedelta(minutes=1)
                elif ret.lower() == 'h':
                    adjust = timedelta(hours=1)
                elif ret.lower() == 'd':
                    adjust = timedelta(days=1)
                if ret.isupper():
                    your_reduction += adjust
                else:
                    your_reduction -= adjust
                your_reduction = min(max_duration, your_reduction)
                your_reduction = max(timedelta(), your_reduction)
            elif ret in 'bB':
                if ret.islower():
                    bias += 1
                else:
                    bias -= 1
                bias = max(min(bias, maxpts-1), -maxpts+1)

        BestOfGame(
            self.state, m,
            functools.partial(self._bet_callback, m, your_reduction, our_reduction),
            start=(max(bias,0), -min(bias,0)),
        )


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
        self.speed, self.bias = 0.55, 0.0

        self.update_msg()
        self.pic(m)

    def update_msg(self):
        self.message = '  ·  '.join(f'{we}–{you}' for we, you in zip(*self.pts))

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
        m.pop(self)

    @bind()
    def pic(self, m):
        cur = self.current
        p = lambda b: max(min((1.020**b) / (1.020**b + 1), 0.93), 0.07)
        conv = lambda p: self.speed * p

        win = random.random() <= conv(p(self.bias) if cur == 0 else 1 - p(self.bias))
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


KEYS = [
    'points',
    'streak',
    'perm_value',
    'added',
    'to_add',
    'last_checkin',
    'permissions',
    'add_next_illegal_mas',
    'scores_you',
    'last_close_unlocked',
]


@click.command('mack-test')
@plugin.db_argument('loader')
def mack_test(loader):
    state = MackerelState(loader)
    state.execute('add_permissions()')


def visible(func):
    func.visible = True
    return func


Wait = namedtuple('Wait', ['end', 'bet'])


class MackerelState:

    def __init__(self, loader):
        self.cfg = loader.cfg['mackerel']
        local = path.join(loader.path, 'mackerel.state')

        if loader.remote:
            remote = path.join(loader.remote, 'mackerel.state')
            rsync_file(remote, local)

        with open(local, 'rb') as f:
            self._state = pickle.load(f)
        # while isinstance(self._state['wait'], Wait):
            # self._state['wait'] = self._state['wait'].end
        # self._state['wait'] = Wait(self._state['wait'], False)

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

    def tick(self):
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
    def nqueued(self):
        return len(self._state['queue'])

    @property
    def waiting(self):
        now = datetime.now()
        if self._state['wait'] and self._state['wait'].end > now:
            return self._state['wait'].end.replace(microsecond=0) - now.replace(microsecond=0)
        self._state['wait'] = None
        return None

    @property
    def closed(self):
        if self.waiting:
            return self._state['wait'].bet

    def mas(self, m, physical=False):
        self.add_permissions(-1, msg=False)
        if physical or (self.permissions >= 0 and not self.blocked):
            m.popup_message('Removed one permission')
        else:
            self.event('illegal-mas', m)

    def game_for(self, m, npts):
        assert not self.waiting
        self.event('game-for', m, npts=npts)

    def game_against(self, m, npts):
        assert not self.waiting
        self.event('game-against', m, npts=npts)

    @visible
    def add_permissions(self, new=1, msg=True):
        self._state['permissions'] = min(self.permissions + new, 1)
        if msg:
            self.m.popup_message(f'Added permission: {new}, now {self.permissions}')

    @visible
    def close_wait(self, msg=True):
        if self._state['wait']:
            self._state['wait'] = Wait(self._state['wait'].end, True)
        if msg:
            self.m.popup_message('Closed wait period')

    @visible
    def add_wait(self, days=0, hours=0, msg=True):
        if not isinstance(days, timedelta):
            duration = timedelta(days=days, hours=hours)
        else:
            duration = days
        if self.waiting:
            if duration > self.waiting:
                bet = False
            else:
                bet = self._state['wait'].bet
            self._state['wait'] = Wait(self._state['wait'].end + duration, bet)
        else:
            self._state['wait'] = Wait(datetime.now() + duration, False)
        if msg:
            self.m.popup_message(f'Waiting {days}d {hours}h')

    @visible
    def nothing(self, msg=True):
        if msg:
            self.m.popup_message('Nothing')


class Mackerel(plugin.PluginBase):

    def __init__(self):
        super().__init__()
        self.commands.append(mack_test)

    def activate(self):
        super().activate()
        self.state = MackerelState(self.loader)
        self.cfg = self.loader.cfg['mackerel']

    @property
    def picker(self):
        with self.loader.database() as db:
            return db.picker(self.cfg['picker'])

    @property
    def has_permission(self):
        return self._permissions > 0

    @property
    def to_add(self):
        return self._to_add - self._added

    def tick(self):
        now = datetime.now()
        passed = (now - self._last_checkin) / timedelta(hours=1)
        self._perm_value += passed * self._points * self.cfg['clock']['per_hour']
        self._last_checkin = now

    def deactivate(self):
        super().deactivate()
        del self.state

    def get_default_program(self):
        return functools.partial(MackerelSlideshow, self.state, self.picker)

    def add_failed(self, filename, reason):
        if reason != 'collision':
            return
        penalty = self.cfg['add']['collision_penalty']
        print(f'[mackerel] penalty of {penalty}')
        self._added -= penalty

    def add_succeeded(self, pic):
        if not self.add_condition(pic):
            return
        if not self._last_close_unlocked:
            print('Add not credited, was not unlocked')
            return

        self._added += 1
        print(f'[mackerel] added: {self._added} / {self._to_add}')
        if self._added < self._to_add:
            return

        new_pts = max(0, self._points - 1)
        print(f'[mackerel] points update: {self._points} -> {new_pts}')
        self._points = new_pts

        factor = random.uniform(*self.cfg['add']['multiplier_range'])
        self._to_add = int(factor * self._to_add)
        self._added = 0
        print(f'[mackerel] to add: {self.to_add}')

    def renew(self, m, npts):
        msg = []
        p = inflect.engine()

        streak_inc = not bool(self._scores_you)
        awins = sum(1 for n in self._scores_you if n > npts)

        msg += [f'Awarded {awins} additional {p.plural("win", awins)}']

        self._add_next_illegal_mas, awins = limiting_sub(self._add_next_illegal_mas, 2, awins)
        msg += [f'After atoning for violations: {awins}']

        if awins > 0:
            prevstreak = self._streak
            self._streak, awins = limiting_sub(self._streak, 0, awins, cost=2, effectiveness=3)
            msg += [f'Streak reduced from {prevstreak} to {self._streak}']
        elif streak_inc and random.random() <= self.cfg['streakprob']:
            self._streak += 1
            msg += [f'Streak increased to {self._streak}']
        else:
            msg += [f'Streak remains at {self._streak}']

        total_wins = 1 + awins
        msg += [f'New permissions: {total_wins}']

        add = sum(range(self._streak, 0, -2))
        msg += [f'Added {add} due to streak']
        npts += add

        msg += [f'New points level {npts}']
        self._points = npts
        self._permissions += total_wins
        self._scores_you = []
        self._added = 0
        self._to_add = self.cfg['add']['num']

        m.popup_message(msg)
