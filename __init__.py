import atexit
from datetime import datetime, timedelta, date
from itertools import groupby
from math import ceil,sqrt
from os.path import join
from random import choice, random
from string import ascii_lowercase

import inflect
import yaml
import numpy as np

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

        perm_cfg = cfg['perm']
        self.perm_we_picker = self.picker(perm_cfg['pickers']['we'])
        self.perm_you_picker = self.picker(perm_cfg['pickers']['you'])
        self.perm_value = lambda pic: pic.eval(perm_cfg['value'])
        self.perm_num = perm_cfg['num']
        self.perm_prob = perm_cfg['prob']
        self.perm_margins = perm_cfg['margins']
        self.perm_break = perm_cfg['break']

        if self.remote:
            remote_status = join(self.remote, 'mackerel.yaml')
            rsync_file(remote_status, self.status_file)

        with open(self.status_file, 'r') as f:
            status = yaml.load(f)
        for k in KEYS:
            setattr(self, '_{}'.format(k), status[k])

    def exit(self):
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

    def give_permission(self, permission, reduced=0):
        if permission:
            self._perm_until = datetime.now() + timedelta(minutes=60-reduced)

    def block_until(self, delta=None):
        if delta is None:
            delta = self.perm_break
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
        if self.leader == leader:
            points += self._streak * (self._streak + 1) // 2
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

butter.db.database_class = Database


class MackerelSlideshow(Slideshow):

    def __init__(self, m):
        super(MackerelSlideshow, self).__init__(m, m.db.leader_picker)
        self.update_msg(m)

    def make_current(self, m):
        self.picker = m.db.leader_picker
        self.update_msg(m)

    @bind()
    def pic(self, m):
        super(MackerelSlideshow, self).pic(m, set_msg=False)
        self.update_msg(m)

    def update_msg(self, m):
        if m.db.leader == 'none':
            self.message = 'Undecided'
            return

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
        elif m.db.we_leading and m.db.can_ask_perm:
            PermGame(m)
        else:
            m.db.popup_message('Nothing to do')

del MackerelSlideshow.keymap['P']
MainWindow.default_program = MackerelSlideshow


class BestOfGame(FromPicker):

    def __init__(self, m):
        super(BestOfGame, self).__init__(m, m.db.picker())

        self.pts = {'we': [0, 0, 0], 'you': [0, 0, 0]}
        self.max_pts = [5, 5, 10]
        self.current = choice(('we', 'you'))
        self.prev_winner = None
        self.speed = 0.55
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
        self.message = '  ·  '.join('{}–{}'.format(we, you)
                                    for we, you in zip(self.pts['we'], self.pts['you']))


class PermGame(FromPicker):

    @staticmethod
    def num_we(dist_we, dist_you, num_you, prob):
        max_num = max(dist_we[-1], dist_you[-1])
        you_cumdist = np.zeros((max_num+1,), dtype=float)
        for n, g in groupby(dist_you):
            you_cumdist[n:] += len(list(g)) / len(dist_you)
        you_cumdist = 1.0 - you_cumdist ** num_you

        we_cumdist = np.zeros((max_num+1,), dtype=float)
        for n, g in groupby(dist_we):
            we_cumdist[n:] += len(list(g)) / len(dist_you)

        def prob_you(num_we):
            we = we_cumdist.copy() ** num_we
            prob = we[0] * you_cumdist[0]
            prob += sum((we[1:] - we[:-1]) * you_cumdist[1:])
            return prob

        minus, plus = 1, num_you
        while prob_you(plus) > prob:
            minus = plus
            plus *= 2
        while plus > minus + 1:
            test = (minus + plus) // 2
            if prob_you(test) > prob:
                minus = test
            else:
                plus = test

        return plus

    def __init__(self, m):
        self.picker_we = m.db.perm_we_picker
        self.picker_you = m.db.perm_you_picker

        dist_we = sorted([m.db.perm_value(p) for p in self.picker_we.get_all()])
        dist_you = sorted([m.db.perm_value(p) for p in self.picker_you.get_all()])
        self.num_we = PermGame.num_we(dist_we, dist_you,
                                        m.db.perm_num,
                                        m.db.perm_prob)

        self.remaining = m.db.perm_num
        self.you_turn = True
        self.total_added = 0
        self.prev_val = 0
        self.you_pts = 0
        self.we_pts = 0

        m.db.block_until()
        super(PermGame, self).__init__(m, self.picker_you)

        m.popup_message(['You get to pick from {}'.format(self.remaining),
                         'We get to pick from {}'.format(self.num_we)])
        self.pic(m)

    def pause(self, m):
        now = datetime.now()
        if hasattr(self, 'until'):
            self.delta_until = self.until - now
            self.delta_before = self.before - now

    def unpause(self, m):
        now = datetime.now()
        if hasattr(self, 'delta_until'):
            self.until = now + self.delta_until
            self.before = now + self.delta_before

    @bind()
    def pic(self, m):
        pic = super(PermGame, self).pic(m, set_msg=False)
        val = m.db.perm_value(pic)
        e = inflect.engine()
        self.remaining -= 1

        if self.you_turn:
            self.you_pts = max(self.you_pts, val)
            self.message = '{} {} remaining after this, {} {} so far'.format(
                self.remaining, e.plural('image', self.remaining),
                self.you_pts, e.plural('point', self.you_pts),
            )
            if self.remaining == 0:
                m.popup_message('You pick {} {}, our turn'.format(
                    self.you_pts, e.plural('point', self.you_pts)
                ))
                self.remaining = self.num_we
                self.you_turn = False
                self.picker = self.picker_we
                self.pic(m)
            return

        msg = '{} {} remaining after this'.format(
            self.remaining, e.plural('image', self.remaining),
        )
        now = datetime.now()
        self.we_pts = max(self.we_pts, val)
        self.prev_val = max(self.prev_val - 1, val, 1)
        if (self.remaining <= 0 and self.prev_val == 1) or val >= self.you_pts:
            self.message = msg
            conf = choice(ascii_lowercase)
            ret = m.popup_message([
                '{} – {}'.format(self.we_pts, self.you_pts),
                'Permission {}'.format('granted' if self.you_pts > self.we_pts else 'denied'),
                'Confirm with {}'.format(conf.upper()),
            ])
            if conf == ret.lower():
                m.db.give_permission(self.you_pts > self.we_pts,
                                     reduced=min(55, self.total_added/3))
                m.db.block_until(self.total_added/4)
            else:
                m.db.block_until(m.db.perm_break + self.total_added/4)
            m.unregister()
            return
        if hasattr(self, 'until') and now < self.until:
            add = int(ceil((self.until - now).total_seconds()))
            msg += '. Added {} {} (too soon).'.format(add, e.plural('point', add))
            self.remaining += add
            self.total_added += add
        elif hasattr(self, 'before') and now > self.before:
            add = int(ceil((now - self.before).total_seconds()))
            msg += '. Added {} {} (too late).'.format(add, e.plural('point', add))
            self.remaining += add
            self.total_added += add
        self.message = msg

        m_until, m_before = m.db.perm_margins
        until = self.prev_val - (1.0 - m_until) * sqrt(self.prev_val)
        before = self.prev_val + (m_before - 1.0) * sqrt(self.prev_val)
        self.until = now + timedelta(seconds=until)
        self.before = now + timedelta(seconds=before)
