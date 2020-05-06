import numpy as np

from keras.utils import Progbar
from keras.utils import OrderedEnqueuer
from keras.utils.generic_utils import to_list


def predict_labels_generator(model, generator,
        steps=None,
        max_queue_size=10,
        workers=1,
        verbose=1):
    """Reimplementation of the Keras function `predict_generator`, to return
    also the labels given by the generator.
    """
    model._make_predict_function()

    steps_done = 0
    all_outs = []
    all_labels = []

    if steps is None:
        steps = len(generator)
    enqueuer = None

    try:
        enqueuer = OrderedEnqueuer(generator)
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        output_generator = enqueuer.get()

        if verbose == 1:
            progbar = Progbar(target=steps)

        while steps_done < steps:
            generator_output = next(output_generator)
            if isinstance(generator_output, tuple):
                # Compatibility with the generators
                # used for training.
                if len(generator_output) == 2:
                    x, y = generator_output
                else:
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y)`. '
                                     'Found: ' +
                                     str(generator_output))
            else:
                raise ValueError('Generator should yield a tuple `(x, y)`')

            outs = model.predict_on_batch(x)
            outs = to_list(outs)
            labels = to_list(y)

            if not all_outs:
                for out in outs:
                    all_outs.append([])

            if not all_labels:
                for lab in labels:
                    all_labels.append([])

            for i, out in enumerate(outs):
                all_outs[i].append(out)

            for i, lab in enumerate(labels):
                all_labels[i].append(lab)

            steps_done += 1
            if verbose == 1:
                progbar.update(steps_done)

    finally:
        if enqueuer is not None:
            enqueuer.stop()

    if len(all_outs) == 1:
        if steps_done == 1:
            all_outs = all_outs[0][0]
            all_labels = all_labels[0][0]
        else:
            all_outs = np.concatenate(all_outs[0])
            all_labels = np.concatenate(all_labels[0])

        return all_outs, all_labels

    if steps_done == 1:
        all_outs = [out[0] for out in all_outs]
        all_labels = [lab[0] for lab in all_labels]
    else:
        all_outs = [np.concatenate(out) for out in all_outs]
        all_labels = [np.concatenate(lab) for lab in all_labels]

    return all_outs, all_labels

