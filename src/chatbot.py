#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

chat_text = ""


def interact_model(
    model_name='117M',
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
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
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
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """
    global chat_text

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [1, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=1,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        while True:
            raw_text = input("Enter a message >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Enter a message >>> ")
            
            chat_text += user_name + ": " + raw_text
            print("="*80)
            print(chat_text)
            print("="*80)
            context_tokens = enc.encode(chat_text)
            
            while True:
                out = sess.run(output, feed_dict={
                    context: [context_tokens]
                })[:, len(context_tokens):]

                text = enc.decode(out[0])
                print("="*40)
                print(text)
                print("="*40)
                response = text[:text.index(user_name + ": ")]
                print('"' + response + '"')
                print(len(response))
                    
                if response != "\n":
                    chat_text += response
                    print(response)
                    break

if __name__ == '__main__':
    fire.Fire(interact_model)
