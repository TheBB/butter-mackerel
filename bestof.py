from random import choice, random

import inflect

from butter.programs import bind, FromPicker


class BestOfGame(FromPicker):

    def __init__(self, m):
        cfg = m['mackerel']['bestof']
        super(BestOfGame, self).__init__(m, m.db.picker(cfg['picker']))

        self.trigger = lambda pic: pic.eval(cfg['trigger'])
        self.value = lambda pic: pic.eval(cfg['value'])

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
        while self.trigger(pic) != win:
            pic = self.picker.get()

        pic = super(BestOfGame, self).pic(m, set_msg=False, pic=pic)

        if not win:
            self.current = self.other(cur)
            return

        npts = self.value(pic)
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
            m.unregister(self, cur)
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
