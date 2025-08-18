from datetime import datetime

from src.NCfold.train_and_test import train_and_test


if __name__ == '__main__':
    print(f'Time: {datetime.now()}')
    train_and_test()
