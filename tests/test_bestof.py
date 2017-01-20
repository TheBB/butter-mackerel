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
        prog.speed = 1.0
        while self.program:
            prog.pic(self)

    def unregister(self, program, winner):
        super(Main, self).unregister(program, winner)
        winners[winner] += 1


if __name__ == '__main__':
    cfg = butter.cfg
    cfg.load_plugin('mackerel')
    with cfg.database(sys.argv[1], write=False) as db:
        for i in range(200):
            Main(db)
            print(i+1, winners, db._bias)
