from random import sample

from data.loader import DataLoader
from system.system import System


class Measurer:
    def measure_accuracy(self, nregistered=40, nunregistered=40):
        s = System()
        registered = sample(list(DataLoader.get_img_data('registered')), nregistered)
        unregistered = sample(list(DataLoader.get_img_data('unregistered')), nunregistered)

        self._test_identification(s, registered)
        self._test_unregistered(s, unregistered)

    def _test_unregistered(self, s, unregistered):
        let_in = 0
        for _, img in unregistered:
            if s.verify(img, '')[0]:
                let_in += 1
        print(f'#unregistered users let in by the system: {let_in}/{len(unregistered)}')

    def _test_identification(self, s, registered):
        successful = 0
        for id, img in registered:
            let_in, identified = s.verify(img, id)
            if let_in and id == identified:
                successful += 1
        print(f'#registered users successfully identified by the system: {successful}/{len(registered)}')


if __name__ == '__main__':
    m = Measurer()
    m.measure_accuracy()
