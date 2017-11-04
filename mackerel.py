from datetime import datetime, timedelta
import functools
import inflect
import numpy as np
from os import path
import random
import string
import yaml

from butter.db import rsync_file
from butter.programs import Slideshow, FromPicker, bind
import butter.plugin as plugin


KEYS = [
    'leader',
    'points',
    'streak',
    'perm_value',
    'added',
    'to_add',
    'last_checkin',
    'permissions',
    'add_next_illegal_mas',
    'scores_you',
]


def sub_value(secs, threshold, limit, spread):
    secs = (secs - threshold) / spread
    h = lambda x: np.exp(-1/x) if x > 0 else 0
    g = h(secs) / (h(secs) + h(1-secs))
    g *= np.arctan(secs) / np.pi + 1/2
    return float(g * limit)


class MackerelSlideshow(Slideshow):

    def __init__(self, mackerel, m):
        self.previous = None
        self.mackerel = mackerel
        self.acc_prob = 1.0
        super().__init__(m, mackerel.picker)
        self.update_msg()

    def make_current(self, m):
        self.picker = self.mackerel.picker
        self.update_msg()

    def update_msg(self):
        mack = self.mackerel
        p = inflect.engine()

        msg = f'{mack._points} {p.plural("point", mack._points)}'
        msg += f' | {mack._permissions} {p.plural("permission", mack._permissions)}'
        msg += f' | add {mack.to_add}'
        msg += f' | clock {mack._perm_value * 100:.2f}'
        msg += f' | acc {(1 - self.acc_prob) * 100:.2f}%'

        self.message = msg

    @bind()
    def pic(self, m):
        mack = self.mackerel
        cfg = mack.cfg['clock']
        pic = super().pic(m, set_msg=False)
        val = mack.clock_value(pic)

        dt = (datetime.now() - self.previous).total_seconds() if self.previous else None
        if val > 0:
            mack._perm_value += val * cfg['pic_multiplier']
        elif dt:
            sub = sub_value(dt, cfg['threshold'], cfg['limit'], cfg['spread'])
            if mack._perm_value > 0.0:
                mack._perm_value = max(0.0, mack._perm_value - sub)
            else:
                prob = min(sub, cfg['max_reduction_probability'])
                self.acc_prob *= 1.0 - prob
                if random.random() < prob:
                    confirm = random.choice(string.ascii_lowercase)
                    ret = m.popup_message([f'Reduction available, confirm with {confirm.upper()}'])
                    if confirm == ret.lower():
                        mack._points = max(0, mack._points - 1)
                        self.acc_prob = 1.0

        self.previous = datetime.now()
        self.update_msg()

    @bind('m')
    def mas(self, m):
        mack = self.mackerel
        if mack.has_permission:
            mack._permissions -= 1
            m.popup_message("Ok, you have permission")
        else:
            mack._points += mack._add_next_illegal_mas
            mack._add_next_illegal_mas += 1
            m.popup_message("Nuh-uh, you don't have permission")
        self.update_msg()

    @bind('M')
    def reg_mas(self, m):
        mack._permissions -= 1
        m.popup_message("Ok, one permission removed")
        self.update_msg()

    @bind('g')
    def bestof(self, m):
        if self.mackerel._points == 0 and self.mackerel._perm_value == 0.0:
            BestOfGame(self.mackerel, m)
        else:
            m.popup_message('Nothing to do')

del MackerelSlideshow.keymap['P']


class BestOfGame(FromPicker):

    def __init__(self, mackerel, m):
        self.mackerel = mackerel

        cfg = mackerel.cfg['bestof']
        super().__init__(m, m.db.picker(cfg['picker']))

        self.trigger = lambda pic: pic.eval(cfg['trigger'])
        self.value = lambda pic: pic.eval(cfg['value'])
        self.cfg = cfg

        self.pts = [[0,0,0], [0,0,0]]
        self.max_pts = [2,2,4]
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

        if winner == 1:
            self.mackerel._scores_you.append(npts)
        else:
            self.mackerel.renew(m, npts)
        m.pop(self, winner)

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


class Mackerel(plugin.PluginBase):

    def activate(self):
        super().activate()
        self.cfg = self.loader.cfg['mackerel']
        self.status_file = path.join(self.loader.path, 'mackerel.yaml')

        self.clock_value = lambda pic: pic.eval(self.cfg['clock']['pic_value'])
        self.add_condition = lambda pic: pic.eval(self.cfg['add']['condition'])

        # if self.loader.remote:
        #     remote_status = path.join(self.loader.remote, 'mackerel.yaml')
        #     rsync_file(remote_status, self.status_file)
        with open(self.status_file, 'r') as f:
            data = yaml.load(f)
        for key in KEYS:
            setattr(self, f'_{key}', data[key])

        now = datetime.now()
        passed = (now - self._last_checkin) / timedelta(hours=1)
        self._perm_value += passed * self._points * self.cfg['clock']['per_hour']
        self._last_checkin = now

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

    def deactivate(self):
        super().deactivate()

        status = {k: getattr(self, '_{}'.format(k)) for k in KEYS}
        with open(path.join(self.loader.path, 'mackerel.yaml'), 'w') as f:
            yaml.dump(status, f, default_flow_style=False)
        if self.loader.remote:
            remote_status = path.join(self.loader.remote, 'mackerel.yaml')
            rsync_file(self.status_file, remote_status)

    def get_default_program(self):
        return functools.partial(MackerelSlideshow, self)

    def add_failed(self, filename, reason):
        if reason != 'collision':
            return
        penalty = self.cfg['add']['collision_penalty']
        print(f'[mackerel] penalty of {penalty}')
        self._added -= penalty

    def add_succeeded(self, pic):
        if not self.add_condition(pic):
            return
        self._added += 1
        if self._added == self._to_add:
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

        additional_wins = sum(1 for n in self._scores_you if n > npts)
        if additional_wins > 0:
            msg += [f'Awarded {additional_wins} additional {p.plural("win", additional_wins)}']
        violations = self._add_next_illegal_mas - 2
        if additional_wins >= violations > 0:
            self._add_next_illegal_mas = 2
            additional_wins -= violations
            msg += [f'Atoned for all {violations} {p.plural("violation", violations)}']
        elif 0 < additional_wins < violations:
            msg += [f'Atoned for {additional_wins} {p.plural("violation", additional_wins)}']
            self._add_next_illegal_mas -= additional_wins
            additional_wins = 0

        total_wins = 1 + additional_wins
        msg += [f'New permissions: {total_wins}']
        if additional_wins == 0:
            add = self._streak * (self._streak + 1) // 2
            npts += self._streak * (self._streak + 1) // 2
            self._streak += 1
            msg += [f'Added {add} due to streak, next time {self._streak + add}']
        else:
            self._streak = 1
            msg += ['Streak vanquished']

        msg += ['New points level {npts}']
        self._points = npts
        self._permissions += total_wins
        self._scores_you = []

        m.popup_message(msg)
