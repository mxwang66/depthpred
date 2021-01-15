class Model(object):
    def __init__(self, hypers):
        self.hypers = hypers

    def make_placeholders():
        raise Exception("Models have to implement make_placeholders!")
    
    def make_weights():
        raise Exception("Models have to implement make_weights!")
    
    def __call__():
        raise Exception("Models have to implement __call__!")

    def make_metrics():
        raise Exception("Models have to implement make_metrics!")

    def set_train_op(self, train_op):
        self.train_op = train_op

