import sys
import yaml
import butter

from bestof import BestOfGame


winners = {
    'you': 0,
    'we': 0,
}


class Main(butter.Main):

    def __init__(self, db):
        super(Main, self).__init__(db)

        prog = BestOfGame(self)
        while self.program:
            prog.pic(self)

    def unregister(self, program, winner):
        super(Main, self).unregister(program, winner)
        winners[winner] += 1


if __name__ == '__main__':
    cfg = butter.cfg
    cfg.load_plugins()
    db = cfg.database(sys.argv[1])
    for i in range(100):
        Main(db)
        print(i+1, winners)
