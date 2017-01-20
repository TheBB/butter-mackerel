from random import choice, random

import inflect

from butter.programs import bind, FromPicker


class BestOfGame(FromPicker):

    def __init__(self, m):
        cfg = m['mackerel']['bestof']
        super(BestOfGame, self).__init__(m, None)

        self.trigger = lambda pic: pic.eval(cfg['trigger'])
        self.value = lambda pic: pic.eval(cfg['value'])
        self.speed = cfg['speed']

        self.pts = {'we': [0, 0, 0], 'you': [0, 0, 0]}
        self.max_pts = [5, 5, 10]
        self.current = choice(('we', 'you'))
        self.prev_winner = None
        self.done = False

        self.update_msg(m)
        self.pic(m)

    @staticmethod
    def other(pl):
        return next(iter({'we', 'you'} - {pl}))

    def add_pts(self, winner, npts):
        l_pts = self.pts[self.other(winner)]
        w_pts = self.pts[winner]
        lost = 0
        while npts > 0 and w_pts[-1] < self.max_pts[-1]:
            i = 0
            w_pts[i] += 1
            while w_pts[i] == self.max_pts[i]:
                if i == len(self.max_pts) - 1:
                    break
                w_pts[i+1] += 1

                l = l_pts[i]
                for j in range(0, i):
                    l *= self.max_pts[j]
                lost += l

                w_pts[i] = l_pts[i] = 0
                i += 1
            npts -= 1

        return lost

    @bind()
    def pic(self, m):
        p = lambda b: max(min((1.020**b) / (1.020**b + 1), 0.93), 0.07)
        current = 'we' if random() <= p(m.db._bias) else 'you'
        win = random() <= self.speed
        pic = m.db.get_picker(current).get()
        while self.trigger(pic) != win:
            pic = m.db.get_picker(current).get()

        super(BestOfGame, self).pic(m, set_msg=False, pic=pic)

        if not win:
            return

        npts = self.value(pic)
        lost = self.add_pts(current, npts)
        self.update_msg(m)
        e = inflect.engine()

        if self.pts[current][-1] == self.max_pts[-1]:
            total = self.max_pts[-1] - self.pts[self.other(current)][-1]
            msg = '{} win with {} {}'.format(
                current.title(), total, e.plural('point', total)
            )
            m.popup_message(msg)
            m.db.update_points_leader(current, total)
            m.unregister(self, current)
            return

        sign = 1 if current == 'we' else -1
        m.db._bias -= sign * min(npts, 15)
        # m.db._bias += sign * lost

        msg = ['{} {} for {}'.format(npts, e.plural('point', npts),
                                     'us' if current == 'we' else 'you')]
        msg.append('{:.2f}%'.format(p(m.db._bias) * 100))
        msg.append('{} points were lost'.format(lost))

        self.update_msg(m)
        m.popup_message(msg)

    def update_msg(self, m):
        self.message = '  ·  '.join('{}–{}'.format(we, you)
                                    for we, you in zip(self.pts['we'], self.pts['you']))
