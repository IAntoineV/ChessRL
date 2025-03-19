import abc


class ChessBot(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_bot_move(self, move):
        pass

    @abc.abstractmethod
    def initialize_pos(self, list_moves):
        pass