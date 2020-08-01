# SMS Organizer

## About
SMS Organizer contains the machine learnt model (and all scripts used to build it) for classifying SMSs into common categories like (personal, important, transactions, advertisements, spam) with an accuracy of 93% for each message. 

However when deployed, it should be used to analyse entire conversations and classify them. Since most conversations contain more than 1 message, real life accuracy jumps to around 99%. 

## Usage
The predict.py script contains the function predict() which takes two inputs, message (string) and time (DD MMM YYYY HH:MM). These two are the main features this model uses to classify the SMSs.

To use, simply call this function and pass in the required parameters to get the label of the SMS for your very own implementation using this model.

## Features Used
The main features used are the message text and the time the SMS was received. 

These are further broken down into trainable features and some other secondary features that were derived from the two main features like presence of URLs, digits etc in the text.

## Model
The trained model can be loaded from SMS_organizer/models/model.h5 and the data that was fed into it is at SMS_organizer/data/train_db/dataset.csv.

You can view it's architecture using tf.keras.utils.plot_model and other such functions. The format of the data can be viewed from dataset.csv.

## Related Projects
* Android App implementing this project : https://github.com/ChiragKalra/sms-organiser-android

* Discord Bot to collect training labels : https://github.com/ChiragKalra/data-label-discord-bot


## Requirements 
* numpy

* pandas

* tensorflow

* Keras

## Owners
Siddhant Sharma: https://github.com/Sids2k

Chirag Kalra: https://github.com/ChiragKalra

## License
Copyright - 2020 - Siddhant Sharma and Chirag Kalra

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
