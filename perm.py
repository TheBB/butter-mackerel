from datetime import datetime, timedelta, date
from itertools import groupby
from math import ceil, floor, sqrt
from random import choice
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
from string import ascii_lowercase

import inflect
import numpy as np

from butter.programs import bind, FromPicker


e = inflect.engine()

class PermGame(FromPicker):

    @staticmethod
    def num_we(dist_we, dist_you, num_you, prob):
        max_num = max(dist_we[-1], dist_you[-1])
        you_cumdist = np.zeros((max_num+1,), dtype=float)
        for n, g in groupby(dist_you):
            you_cumdist[n:] += len(list(g)) / len(dist_you)
        you_cumdist = 1 - you_cumdist ** num_you

        we_cumdist = np.zeros((max_num+1,), dtype=float)
        for n, g in groupby(dist_we):
            we_cumdist[n:] += len(list(g)) / len(dist_we)

        def prob_you(num_we):
            we = we_cumdist.copy() ** num_we
            we = np.hstack(([0], we))
            prob = sum((we[1:] - we[:-1]) * you_cumdist)
            return prob

        minus, plus = 1, num_you
        while prob_you(plus) > prob:
            minus = plus
            plus *= 2
        while plus - minus > 1e-4:
            test = (minus + plus) / 2
            if prob_you(test) > prob:
                minus = test
            else:
                plus = test

        return int(ceil(plus))

    @staticmethod
    def winrate(D, U, Q, N=5000):
        ri, ci, data = [], [], []

        for i in range(0, N+1):
            ri.append(max(i-D, 0))
            ci.append(i)
            data.append(i/N)

            ri.append(i)
            ci.append(i)
            data.append(-1)

            ri.append(min(i+U, N))
            ci.append(i)
            data.append(1-i/N)

            ri.append(N+1)
            ci.append(i)
            data.append(1)

        mx = csc_matrix((data, (ri, ci)), shape=(N+2, N+1))
        dist, *_ = lsqr(mx, np.hstack((np.zeros(N+1), 1)))
        return sum(i/N * d for i, d in enumerate(dist[:Q]))

    @staticmethod
    def find_U_and_threshold(down, rrate, wrate, N=5000):
        D = int(floor(N*down))
        U = int(ceil(rrate / (1 - rrate) * D))

        Qmin = 0
        Qmax = N

        while Qmax - Qmin > 1:
            Q = Qmin + (Qmax - Qmin) // 2
            c_wrate = PermGame.winrate(D, U, Q, N)
            if c_wrate < wrate:
                Qmin = Q
            else:
                Qmax = Q

        return U/N, Qmin/N

    def __init__(self, m, timing=True, complete=False):
        self.timing = timing
        self.complete = complete

        cfg = m['mackerel']['perm']
        self.sub_win = cfg['sub_win']
        self.picker_we = m.db.picker(cfg['pickers']['we'])
        self.picker_you = m.db.picker(cfg['pickers']['you'])
        self.value = lambda pic: pic.eval(cfg['value'])
        self.remaining = cfg['num']
        self.margins = cfg['margins']
        self.brk = cfg['break']
        self.penalty = cfg['penalty']

        self.add_loss, sq = PermGame.find_U_and_threshold(self.sub_win, cfg['redrate'], cfg['winrate'])
        self.qualified = m.db._perm_prob < sq

        dist_we = sorted([self.value(p) for p in self.picker_we.get_all()])
        dist_you = sorted([self.value(p) for p in self.picker_you.get_all()])
        self.num_we = PermGame.num_we(dist_we, dist_you, self.remaining, m.db._perm_prob)

        self.turn = 'you'
        self.pts = {'you': 0, 'we': 0}
        self.total_added = 0
        self.prev_val = 0

        m.db.block_until()
        super(PermGame, self).__init__(m, self.picker_you)

        m.popup_message(['You get to pick from {}'.format(self.remaining),
                         'We get to pick from {}'.format(self.num_we),
                         'Probability {:0.2f}%'.format(m.db._perm_prob * 100),
                         'Qualified' if self.qualified else 'Not qualified',
                         '(Threshold {:0.2f}%)'.format(sq * 100),
                         'Next: {:0.2f}% or {:0.2f}%'.format(
                             (m.db._perm_prob - self.sub_win) * 100,
                             (m.db._perm_prob + self.add_loss) * 100
                         )])
        self.pic(m)

    @property
    def other(self):
        return next(iter({'we', 'you'} - {self.turn}))

    @property
    def your_turn(self):
        return self.turn == 'you'

    @property
    def our_turn(self):
        return self.turn == 'we'

    @property
    def msg_remaining(self):
        return '{} {} remaining after this'.format(
            max(self.remaining, 0), e.plural('image', self.remaining)
        )

    @property
    def msg_pts(self):
        pts = self.pts[self.turn]
        return '{} {} so far'.format(pts, e.plural('point', pts))

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

    def pic_you(self, m):
        self.message = ', '.join((self.msg_remaining, self.msg_pts))
        if self.remaining == 0:
            m.popup_message('You pick {}, our turn'.format(self.msg_pts))
            self.remaining = self.num_we
            self.picker = self.picker_we
            self.turn = 'we'
            self.pic(m)

    @bind()
    def pic(self, m):
        now = datetime.now()
        pic = super(PermGame, self).pic(m, set_msg=False)
        val = self.value(pic)
        self.pts[self.turn] = max(self.pts[self.turn], val)
        self.remaining -= 1

        if self.your_turn:
            self.pic_you(m)
            return

        if self.complete:
            done = self.remaining <= 0
        else:
            done = (self.remaining <= 0 and self.prev_val <= 1) or val >= self.pts['you']
        if done:
            granted = self.pts['you'] > self.pts['we']
            if granted:
                m.db._perm_prob -= self.sub_win
            else:
                m.db._perm_prob += self.add_loss
            m.db._perm_prob = max(0.01, min(0.99, m.db._perm_prob))
            granted = granted and self.qualified
            self.message = self.msg_remaining
            conf = choice(ascii_lowercase)
            ret = m.popup_message([
                '{} – {}'.format(self.pts['we'], self.pts['you']),
                'Permission {}'.format('granted' if granted else 'denied'),
                'Qualified' if self.qualified else 'Not qualified',
                'Confirm with {}'.format(conf.upper()),
            ])
            if conf == ret.lower():
                m.db.give_permission(granted, reduced=min(55, self.total_added/3))
                m.db.block_until(self.total_added/5)
            else:
                m.db.block_until(self.brk + self.total_added/5)
            m.unregister(self, 'you' if granted else 'we', self.pts['you'], self.pts['we'])
            return

        self.prev_val = max(val, self.prev_val - 1, 1)
        add = max(
            (self.until - now).total_seconds() if hasattr(self, 'until') else 0.0,
            (now - self.before).total_seconds() if hasattr(self, 'before') else 0.0,
            0.0,
        )
        add = int(ceil(add / self.penalty))
        add_msg = ''
        if add > 0:
            add_msg = '. Added {} {}.'.format(add, e.plural('point', add))
        self.remaining += add
        self.total_added += add

        self.message = self.msg_remaining + add_msg

        if self.timing:
            m_until, m_before = self.margins
            until = self.prev_val - (1.0 - m_until) * sqrt(self.prev_val)
            before = self.prev_val + (m_before - 1.0) * sqrt(self.prev_val)
            self.until = now + timedelta(seconds=until)
            self.before = now + timedelta(seconds=before)
