import sys
import butter

from slideshow import try_perm, add_value


class FakeDB:

    def __init__(self, db):
        self._perm_value = 0.0
        self.has_perm = False
        self.we_leading = True
        self.picker = db._pickers['we']
        self.sub_value = db.sub_value

if __name__ == '__main__':
    cfg = butter.cfg
    cfg.load_plugin('mackerel')
    with cfg.database(sys.argv[1], write=False) as db:
        fakedb = FakeDB(db)
        wins, tries, subs = 0, 0, 0
        secs = float(sys.argv[2])
        cc = db.cfg['mackerel']['perm']
        add_per_pic = add_value(secs, cc['threshold'], cc['limit'], cc['spread'])
        sub_per_pic = 0.0
        for pic, freq in fakedb.picker.get_dist():
            sub_per_pic += db.sub_value(pic) * cc['sub_mult'] * freq
        if sub_per_pic > add_per_pic:
            print('Too short', sub_per_pic, add_per_pic)
        else:
            while True:
                pic = fakedb.picker.get()
                tries += 1
                if try_perm(pic, fakedb, db.cfg['mackerel']['perm'], secs):
                    wins += 1
                if fakedb._perm_value < 0.0:
                    subs += 1
                print(tries, 'Seconds per win: {:.2f}, time spent below: {:.2f}%'.format(
                    (tries / wins * secs) if wins > 0 else 1e99,
                    subs / tries * 100,
                ))
