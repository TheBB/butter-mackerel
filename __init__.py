from butter import Main
import butter.db

from mackerel.slideshow import MackerelSlideshow
from mackerel.db import Database


def enable():
    butter.db.database_class = Database
    Main.default_program = MackerelSlideshow
