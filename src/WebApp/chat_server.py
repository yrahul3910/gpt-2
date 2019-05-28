#!/usr/bin/python3

import json
import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, Response

# Custom module imports
import model
import sample
import encoder

app = Flask(__name__)


def generate_response(
    chat_text,
    model_name='345M',
    length=128,
    temperature=0.9,
    top_k=40,
    top_p=0.0,
    user_name="Rahul Yedida",
    friend_name="Bhavana",
    filename="yrahul_kvbhavana"
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
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
    # Get encoder and hyper-parameters for model
    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    # Get sample lengths
    if length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" %
                         hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        # Batch size is 1, so use [1, None] shape.
        context = tf.placeholder(tf.int32, [1, None])

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
        # First, we need to write a "checkpoint" file that tells the
        # code what files to use.
        with open("models/{0}/checkpoint".format(model_name), "w") as f:
            f.write('model_checkpoint_path: "{0}"\n'
                    'all_model_checkpoint_paths: "{0}"'.format(filename))

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

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
                return response


@app.route("/api/response", methods=["POST"])
def get_response():
    """
    Gets a response, given the current full chat. Takes a JSON
    parameter:
    {
        user_name: (str) The user's name
        friend_name: (str) The friend's name
        model_name: (str) One of "345M", "117M"
        chat_text: (str) Chat text so far
    }
    """
    data = request.get_json()
    user_name = data['user_name']
    friend_name = data['friend_name']
    model_name = data['model_name']
    chat_text = data['chat_text']

    # Implement fetching filename later.
    response = generate_response(chat_text,
                                 model_name=model_name,
                                 user_name=user_name,
                                 friend_name=friend_name,
                                 filename="yrahul_kvbhavana")

    return Response(json.dumps({"response": response}),
                    status=200,
                    mimetype="application/json")
