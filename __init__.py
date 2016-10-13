from butter.programs import Slideshow, bind
from butter.gui import MainWindow


class RestrictedSlideshow(Slideshow):

    def __init__(self, m):
        cfg = m.db.cfg['mackerel']
        self.leader = cfg['leader']
        self.pickers = {
            key: m.db.picker(val)
            for key, val in cfg['pickers'].items()
        }

        super(RestrictedSlideshow, self).__init__(m, self.pickers[self.leader])
        self.message = self.leader.title()

    @bind()
    def pic(self, m):
        super(RestrictedSlideshow, self).pic(m, set_msg=False)

del RestrictedSlideshow.keymap['P']
MainWindow.default_program = RestrictedSlideshow
