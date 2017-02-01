from datetime import datetime, timedelta, date
from os.path import join
from butter.db import database_class, loader_class, rsync_file

import inflect
import yaml


KEYS = [
    'leader',
    'points',
    'streak',
    'next_mas_add',
    'perm_until',
    'perm_prob',
    'ask_blocked_until',
    'added',
]


class DatabaseLoader(loader_class):

    def add_pic(self, fn, pic, db):
        super(DatabaseLoader, self).add_pic(fn, pic, db)
        db.pic_add_hook(pic)


class Database(database_class):

    def __init__(self, *args, write=True, **kwargs):
        super(Database, self).__init__(*args, **kwargs)
        self.status_file = join(self.path, 'mackerel.yaml')
        self.write = write

        cfg = self.cfg['mackerel']
        self.__pickers = {
            key: self.picker(val)
            for key, val in cfg['pickers'].items()
        }
        self.brk = cfg['perm']['break']
        self.add_cond = lambda pic: pic.eval(cfg['perm']['add_cond'])
        self.add_num = cfg['perm']['add_num']

        if self.remote:
            remote_status = join(self.remote, 'mackerel.yaml')
            rsync_file(remote_status, self.status_file)

        with open(self.status_file, 'r') as f:
            status = yaml.load(f)
        for k in KEYS:
            setattr(self, '_{}'.format(k), status[k])

        self.msg = None

    def close(self):
        super(Database, self).close()
        if not self.write:
            return

        status = {k: getattr(self, '_{}'.format(k)) for k in KEYS}
        with open(self.status_file, 'w') as f:
            yaml.dump(status, f, default_flow_style=False)
        if self.remote:
            remote_status = join(self.remote, 'mackerel.yaml')
            rsync_file(self.status_file, remote_status)

    @property
    def leader_picker(self):
        return self.__pickers[self.leader]

    @property
    def leader(self):
        return self._leader

    @property
    def points(self):
        return self._points

    @property
    def signed_points(self):
        return -self.points if self.you_leading else self.points

    @property
    def we_leading(self):
        return self.leader == 'we'

    @property
    def you_leading(self):
        return self.leader == 'you'

    @property
    def has_perm(self):
        return self._perm_until > datetime.now()

    @property
    def perm_mins(self):
        return (self._perm_until - datetime.now()).seconds // 60

    @property
    def can_ask_perm(self):
        return self._ask_blocked_until <= datetime.now()

    @property
    def mins_until_can_ask_perm(self):
        return (self._ask_blocked_until - datetime.now()).seconds // 60 + 1

    def pic_add_hook(self, pic):
        if self.leader == 'we' and self.add_cond(pic):
            self._added += 1
            if self._added >= self.add_num:
                self._added -= self.add_num
                self.give_permission(True)

    def give_permission(self, permission, reduced=0):
        if permission:
            self._perm_until = datetime.now() + timedelta(minutes=60-reduced)

    def block_until(self, delta=None):
        if delta is None:
            delta = self.brk
        self._ask_blocked_until = datetime.now() + timedelta(minutes=delta)

    def update_points(self, new=None, delta=None, sdelta=None):
        if new is not None:
            self._points = new
            return
        if delta is not None:
            self._points += delta
            self._points = max(self._points, 0)
            return
        sdelta = -sdelta if self.you_leading else sdelta
        self._points += sdelta
        self._points = max(self._points, 0)

    def update_points_leader(self, leader, points):
        assert leader in {'we', 'you'}
        if self.leader == leader == 'we':
            points += self._streak * (self._streak + 1) // 2
            self._streak += 1
        elif self.leader == leader == 'you':
            self._streak += 1
            points = self._streak * (self._streak + 1) // 2
        else:
            self._streak = 1
            if leader == 'you':
                points = 1
        self._leader = leader
        self._points = points
        self._next_mas_add = 0

    def mas(self, skip):
        chg = 0

        if self.you_leading and self.points > 0:
            pos = 'You are leading'
            chg = -2 if skip else -1
        elif self.you_leading:
            pos = 'Figure out something to do here'
        elif self.we_leading:
            if self._perm_until >= datetime.now():
                pos = 'You have permission'
                chg = -1 if skip else self._next_mas_add
                if not skip:
                    self._next_mas_add += 1
                self._perm_until = datetime.now() - timedelta(hours=2)
            elif not skip:
                pos = "You don't have permission"
                chg = 2 * (self._next_mas_add + 1)
                self._ask_blocked_until = datetime.now() + timedelta(hours=1)
                self._next_mas_add += 1
            else:
                return "That doesn't make sense"

        self.update_points(delta=chg)
        return '{}. Delta {:+}. New {}.'.format(pos, chg, self.points)
