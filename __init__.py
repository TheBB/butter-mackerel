import atexit
from datetime import datetime, timedelta, date
from os.path import join
from random import choice, random

import yaml
import inflect

from butter.programs import FromPicker, Slideshow, bind
from butter.gui import MainWindow
import butter.db
from butter.db import database_class, rsync_file


KEYS = [
    'leader',
    'points',
    'streak',
    'last_checkin',
    'last_mas',
    'next_mas_add',
    'perm_until',
    'ask_blocked_until',
]

class Database(database_class):

    def __init__(self, *args, **kwargs):
        super(Database, self).__init__(*args, **kwargs)
        self.status_file = join(self.path, 'mackerel.yaml')
        atexit.register(self.exit)

        cfg = self.cfg['mackerel']
        self.__pickers = {
            key: self.picker(val)
            for key, val in cfg['pickers'].items()
        }

        bestof_cfg = cfg['bestof']
        self.bestof_picker = self.picker(bestof_cfg['picker'])
        self.bestof_trigger = lambda pic: pic.eval(bestof_cfg['trigger'])
        self.bestof_value = lambda pic: pic.eval(bestof_cfg['value'])

        # if self.remote:
        #     remote_status = join(self.remote, 'mackerel.yaml')
        #     rsync_file(remote_status, self.status_file)

        with open(self.status_file, 'r') as f:
            status = yaml.load(f)
        for k in KEYS:
            setattr(self, '_{}'.format(k), status[k])

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
        if self.leader == leader:
            points += self._streak * (self._streak + 1) / 2
            self._streak += 1
        else:
            self._streak = 0
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
                self._last_mas = date.today()
            elif not skip:
                pos = "You don't have permission"
                chg = 2 * (self._next_mas_add + 1)
                self._ask_blocked_until = datetime.now() + timedelta(hours=1)
                self._next_mas_add += 1
            else:
                return "That doesn't make sense"

        self.update_points(delta=chg)
        return '{}. Delta {:+}. New {}.'.format(pos, chg, self.points)

    def exit(self):
        status = {k: getattr(self, '_{}'.format(k)) for k in KEYS}
        with open(self.status_file, 'w') as f:
            yaml.dump(status, f, default_flow_style=False)
        if self.remote:
            remote_status = join(self.remote, 'mackerel.yaml')
            rsync_file(self.status_file, remote_status)

butter.db.database_class = Database


class MackerelSlideshow(Slideshow):

    def __init__(self, m):
        super(MackerelSlideshow, self).__init__(m, m.db.leader_picker)
        self.update_msg(m)

    def make_current(self, m):
        self.picker = m.db.leader_picker

    @bind()
    def pic(self, m):
        super(MackerelSlideshow, self).pic(m, set_msg=False)

    def update_msg(self, m):
        if m.db.leader == 'none':
            self.message = 'Undecided'
        else:
            p = inflect.engine()
            msg = '{} are in the lead with {} {}'.format(
                m.db.leader.title(),
                m.db.points,
                p.plural('point', m.db.points),
            )

            if m.db.we_leading:
                msg += '. '
                if m.db.has_perm:
                    msg += 'Permission for {} minutes.'.format(m.db.perm_mins)
                elif m.db.can_ask_perm:
                    msg += 'Can ask permission.'
                else:
                    msg += 'Can ask permission in {} minutes'.format(m.db.mins_until_can_ask_perm)

            self.message = msg

    @bind('m')
    def mas_noskip(self, m):
        m.popup_message(m.db.mas(skip=False))
        self.update_msg(m)

    @bind('n')
    def mas_skip(self, m):
        m.popup_message(m.db.mas(skip=True))
        self.update_msg(m)

    @bind('g')
    def game(self, m):
        if m.db.points == 0:
            BestOfGame(m)
        elif m.db.you_leading:
            print('nothing')
        elif not m.db.can_ask_perm:
            print('nothing')
        else:
            print('permgame')


class BestOfGame(FromPicker):

    def __init__(self, m):
        super(BestOfGame, self).__init__(m, m.db.picker())

        self.pts = {'we': [0, 0, 0], 'you': [0, 0, 0]}
        self.max_pts = [5, 5, 10]
        self.current = choice(('we', 'you'))
        self.prev_winner = None
        self.speed = 0.3
        self.bias = 0.0
        self.done = False

        self.update_msg(m)
        self.pic(m)

    @staticmethod
    def other(pl):
        return next(iter({'we', 'you'} - {pl}))

    def add_pts(self, winner, npts):
        l_pts = self.pts[self.other(winner)]
        w_pts = self.pts[winner]
        while npts > 0 and w_pts[-1] < self.max_pts[-1]:
            i = 0
            w_pts[i] += 1
            while w_pts[i] == self.max_pts[i]:
                if i == len(self.max_pts) - 1:
                    break
                w_pts[i+1] += 1
                w_pts[i] = l_pts[i] = 0
                i += 1
            npts -= 1

    @bind()
    def pic(self, m):
        cur = self.current
        p = lambda b: max(min((1.020**b) / (1.020**b + 1), 0.93), 0.07)
        conv = lambda p: self.speed * p
        threshold = conv(p(self.bias) if cur == 'we' else 1 - p(self.bias))
        win = random() <= threshold
        pic = self.picker.get()
        while m.db.bestof_trigger(pic) != win:
            pic = self.picker.get()

        pic = super(BestOfGame, self).pic(m, set_msg=False, pic=pic)

        if not win:
            self.current = self.other(cur)
            return

        npts = m.db.bestof_value(pic)
        self.add_pts(cur, npts)
        self.update_msg(m)
        e = inflect.engine()

        if self.pts[cur][-1] == self.max_pts[-1]:
            total = self.max_pts[-1] - self.pts[self.other(cur)][-1]
            msg = '{} win with {} {}'.format(
                cur.title(), total, e.plural('point', total)
            )
            m.popup_message(msg)
            m.db.update_points_leader(cur, total)
            m.unregister()
            return

        sign = 1 if cur == 'we' else -1
        if self.prev_winner != cur:
            self.bias = 0.0
        else:
            self.bias += sign * min(npts, 15)

        msg = ['{} {} for {}'.format(npts, e.plural('point', npts),
                                     'us' if cur == 'we' else 'you')]

        p_t, p_f = conv(p(self.bias)), conv(1 - p(self.bias))
        denom = p_t + p_f - p_t * p_f
        if cur == 'we':
            p_t /= denom
        else:
            p_t = 1 - p_f / denom
        p_f = 1 - p_t
        msg.append('{:.2f}% – {:.2f}%'.format(p_t*100, p_f*100))

        self.prev_winner = cur

        self.update_msg(m)
        m.popup_message(msg)

    def update_msg(self, m):
        self.message = 'we ({}) — ({}) you'.format(
            ', '.join(str(s) for s in self.pts['we']),
            ', '.join(str(s) for s in self.pts['you'][::-1]),
        )

del MackerelSlideshow.keymap['P']
MainWindow.default_program = MackerelSlideshow
