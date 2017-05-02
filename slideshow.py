from datetime import datetime, timedelta, date
import inflect
import numpy as np
from random import choice, random
from string import ascii_lowercase

from butter.programs import bind, Slideshow

from mackerel.bestof import BestOfGame


def add_value(secs, threshold, limit, spread):
    secs = (secs - threshold) / spread
    h = lambda x: np.exp(-1/x) if x > 0 else 0
    g = h(secs) / (h(secs) + h(1-secs))
    g *= np.arctan(secs) / np.pi + 1/2
    return float(g * limit)


def try_perm(pic, db, cfg, dt):
    val = db.sub_value(pic)
    if val > 0:
        db._perm_value -= val * cfg['sub_mult']
    elif dt and not db.has_perm:
        add = add_value(dt, cfg['threshold'], cfg['limit'], cfg['spread'])
        prob = min(add, cfg['perm_prob'])
        if db._perm_value < 0.0:
            db._perm_value = min(0.0, db._perm_value + add)
        elif random() < prob:
            return True
    return False


class MackerelSlideshow(Slideshow):

    def __init__(self, m):
        self.previous = None
        super(MackerelSlideshow, self).__init__(m, m.db.leader_picker)
        self.update_msg(m)

        if m.db.msg:
            m.popup_message(m.db.msg)

    def make_current(self, m):
        self.picker = m.db.leader_picker
        self.update_msg(m)

    @bind()
    def pic(self, m):
        pic = super(MackerelSlideshow, self).pic(m, set_msg=False)
        if not m.db.we_leading:
            self.update_msg(m)
            return
        if self.previous and try_perm(pic, m.db, m['mackerel']['perm'],
                                      (datetime.now() - self.previous).total_seconds()):
            conf = choice(ascii_lowercase)
            ret = m.popup_message([
                'Permission available, confirm with {}'.format(conf.upper()),
            ])
            if conf == ret.lower():
                m.db.give_permission(True)
                self.update_msg(m)
        self.previous = datetime.now()
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
            else:
                msg += ' (' + ', '.join([
                    'To add: {}'.format(m.db.add_num - m.db._added),
                    'sub: {:.2f}'.format(abs(m.db._perm_value) * 100),
                ]) + ')'

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
        else:
            m.popup_message('Nothing to do')

del MackerelSlideshow.keymap['P']
