from datetime import datetime, timedelta, date
from os.path import join
from butter.db import database_class, loader_class, rsync_file

import inflect
import yaml


KEYS = [
    'leader',
    'points',
    'streak',
    'perm_value',
    'added',
    'last_checkin',
    'permissions',
    'add_next_illegal_mas',
]


class DatabaseLoader(loader_class):

    def add_pic(self, fn, pic, db):
        super(DatabaseLoader, self).add_pic(fn, pic, db)
        db.pic_add_hook(pic)


class Database(database_class):

    def __init__(self, *args, write=True, **kwargs):
        self.regular = kwargs['regular']
        super(Database, self).__init__(*args, **kwargs)
        self.status_file = join(self.path, 'mackerel.yaml')
        self.write = write

        cfg = self.cfg['mackerel']
        self._pickers = {
            key: self.picker(val)
            for key, val in cfg['pickers'].items()
        }
        self.add_cond = lambda pic: pic.eval(cfg['perm']['add_cond'])
        self.add_num = cfg['perm']['add_num']
        self.sub_value = lambda pic: pic.eval(cfg['perm']['sub_value'])

        if self.remote:
            remote_status = join(self.remote, 'mackerel.yaml')
            rsync_file(remote_status, self.status_file)

        with open(self.status_file, 'r') as f:
            status = yaml.load(f)
        for k in KEYS:
            setattr(self, '_{}'.format(k), status[k])

        if self.regular:
            passed = (datetime.now() - self._last_checkin) / timedelta(hours=1)
            # self._perm_value -= passed * cfg['perm']['per_hour'] *  + cfg['perm']['per_start']
            self._perm_value -= passed * self._points / 100
            self._last_checkin = datetime.now()

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
        return self._pickers[self.leader]

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
        return self._permissions > 0

    def pic_add_hook(self, pic):
        if self.leader == 'we' and self.add_cond(pic) and self._perm_value >= 0:
            self._added += 1
            if self._added >= self.add_num:
                self._added -= self.add_num
                self.give_permission(True)

    def give_permission(self, permission, reduced=0):
        self._permissions += 1

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
            points = self._streak
        else:
            self._streak = 1
            if leader == 'you':
                points = 1
                self._add_next_illegal_mas = 2
        self._leader = leader
        self._points = points

    def mas(self):
        chg = 0

        if self.you_leading and self.points > 0:
            pos = 'OK, you are leading.'
            chg = -1
        elif self.you_leading:
            pos = '???'
        elif self.we_leading and self.has_perm:
            pos = 'OK, you have permission.'
            chg = 0
            self._permissions -= 1
        else:
            pos = "Nuh-uh. You don't have permission"
            chg = self._add_next_illegal_mas
            self._add_next_illegal_mas += 1

        self.update_points(delta=chg)
        return pos
