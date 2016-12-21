from butter import Main
import butter.db

from mackerel.slideshow import MackerelSlideshow
from mackerel.db import DatabaseLoader, Database


def enable():
    butter.db.database_class = Database
    butter.db.loader_class = DatabaseLoader
    Main.default_program = MackerelSlideshow
