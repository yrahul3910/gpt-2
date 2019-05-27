#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

# Custom module imports
import model
import sample
import encoder

# Total chat so far
chat_text = ""


def interact_model(
    model_name='345M',
    seed=None,
    length=128,
    temperature=0.9,
    top_k=40,
    top_p=0.0,
    user_name="Rahul Yedida",
    friend_name="Bhavana"
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to
     reproduce results
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus
     sampling, overriding top_k if set to a value > 0. A good setting is 0.9.
    """
    global chat_text

    # Get encoder and hyper-parameters for model
    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    # Get sample lengths
    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" %
                         hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        # Batch size is 1, so use [1, None] shape.
        context = tf.placeholder(tf.int32, [1, None])

        # Set up seeds for numpy and tf for reproducible results
        np.random.seed(seed)
        tf.set_random_seed(seed)

        # Get graph to run from sampler using hyper-parameters
        output = sample.sample_sequence(
            hparams=hparams,
            length=length,
            context=context,
            batch_size=1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        # Load checkpoint
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        try:
            while True:
                # Get user message
                raw_text = input("Enter a message >>> ")
                while not raw_text:
                    print('Prompt should not be empty!')
                    raw_text = input("Enter a message >>> ")

                # Update global chat variable
                chat_text += user_name + ": " + raw_text

                # Print out conversation so far
                print("=" * 34 + "CONVERSATION" + "=" * 34)
                print(chat_text)
                print("=" * 80)

                # Encode chat so far
                context_tokens = enc.encode(chat_text)

                while True:
                    # Get samples from sampler graph
                    out = sess.run(output, feed_dict={
                        context: [context_tokens]
                    })[:, len(context_tokens):]

                    # Decode text, and get response
                    # The response is obtained by searching for the user's
                    # name, and fetching all the text till there, which
                    # presumably will be only by the friend
                    text = enc.decode(out[0])
                    response = text[:text.index(user_name + ": ")]

                    # Sometimes, produces empty response because the samples
                    # are only for the user's side
                    if response != "\n":
                        # If a proper response (hopefully!), append to our
                        # global variable and print it out. We don't need to
                        # resample (which we keep doing in case of empty
                        # response
                        chat_text += response
                        print(response)
                        break
        except KeyboardInterrupt:
            print("Conversation ended.")


if __name__ == '__main__':
    fire.Fire(interact_model)
