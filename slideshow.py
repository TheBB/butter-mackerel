import inflect

from butter.programs import bind, Slideshow

from mackerel.bestof import BestOfGame
from mackerel.perm import PermGame


class MackerelSlideshow(Slideshow):

    def __init__(self, m):
        super(MackerelSlideshow, self).__init__(m, m.db.leader_picker)
        self.update_msg(m)

        if m.db.msg:
            m.popup_message(m.db.msg)

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
            m.popup_message('Nothing to do')

del MackerelSlideshow.keymap['P']
