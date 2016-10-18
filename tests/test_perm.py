import sys
import yaml
import butter

from perm import PermGame


winners = {
    'you': 0,
    'we': 0,
}


class Main(butter.Main):

    def __init__(self, db):
        super(Main, self).__init__(db)

        prog = PermGame(self, timing=False, complete=True)
        while self.program:
            prog.pic(self)

    def popup_message(self, msg, align='center'):
        return 'c'

    def unregister(self, program, winner, you_pts, we_pts):
        super(Main, self).unregister(program, winner)
        winners[winner] += 1


if __name__ == '__main__':
    cfg = butter.cfg
    cfg.load_plugins()
    db = cfg.database(sys.argv[1])
    i = 1
    while True:
        Main(db)
        print('{}: we {:0.4f} you {:0.4f}'.format(
            i, winners['we'] / i, winners['you'] / i
        ))
        i += 1
