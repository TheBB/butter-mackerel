from datetime import datetime, timedelta
import functools
import inflect
import numpy as np
from os import path
import random
import string
import yaml

from butter.db import rsync_file
from butter.programs import Slideshow, bind
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
        super().__init__(m, mackerel.picker)
        self.update_msg(m)

    def make_current(self, m):
        self.picker = mackerel.picker
        self.update_msg(m)

    def update_msg(self, m):
        mack = self.mackerel
        p = inflect.engine()

        msg = f'{mack._points} {p.plural("point", mack._points)}'
        if mack.has_permission:
            msg += f' | {mack._permissions} {p.plural("permission", mack._permissions)}'
        msg += f' | add {mack.to_add}'
        msg += f' | clock {mack._perm_value * 100:.2f}'

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
            elif random.random() < min(sub, cfg['max_reduction_probability']):
                confirm = random.choice(string.ascii_lowercase)
                ret = m.popup_message([f'Reduction available, confirm with {confirm.upper()}'])
                if conf == ret.lower():
                    mack._points = max(0, mack._points - 1)

        self.previous = datetime.now()
        self.update_msg(m)

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
        self.update_msg(m)

del MackerelSlideshow.keymap['P']


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
