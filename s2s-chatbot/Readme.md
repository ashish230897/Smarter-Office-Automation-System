This is the code for the sequence -2 - sequence +rule based chatbot.

Functionality:

We believe that a chatbot can help tremendously in the office automation space, save costs and
save countless human hours that can be utilized for more productive work. Therefore we
introduce our chatbot V, a polite, obedient and hilarious chatbot. He can do the following
functions:
1) It can greet users, handle basic conversational context, tell jokes, and is highly
customizable for the needs of any organization to perform a wide range of HR operations
seamlessly, and with minimal changes.

2) It has special authentication parameters built-in, that restrict the access the data access
to unauthorized personnel. For example:
The managers can check attendance and number of hours clocked-in of any day of any
employee by simply typing in the name of the employee, thus saving costs and
increasing productivity.

3) Also due to the chat-app interface, alerts are sent to the manager on a regular basis,
notifying him of the other important statistics of the day like crowd count, crowd density
in different locations, number of employees who clocked in late etc.

4) It can automate several low level HR features for the employees like scheduling
appointments, checking attendance and checking project status. It also helps the
employees learn more about their projects with helpful links.

5) With reference to our lab, at any point in time, we can predict the presence of sir in the
lab, by using an artificial neural network fed with the details of his presence for a time
period of 2 weeks.
The chatbot presented in this project combines traditional rule-based chatbot models with
advanced deep learning models.
Deep Learning models, as amazing as they are, have trouble doing basic tasks like
interpretations of calculation-based questions, or telling novel data that it has not been trained
on, like telling a joke for example, or even telling the time or remembering a name or context for
extended sessions of time.
To overcome these shortcomings, we have integrated a rule-based approach with the
deep-learning models, to create a novel architecture.
The deep-learning model used in this project is sequence-to-sequence model which consists of
two RNNs - an encoder and a decoder. The encoder reads the input sequence, word by word
and emits a context (a function of final hidden state of encoder), which would ideally capture the
essence (semantic summary) of the input sequence. Based on this context, the decoder
generates the output sequence, one word at a time while looking at the context and the previous
word during each timestep.

Recurrent Neural Networks(RNNs), are a special kind of neural network that are capable of
dealing with sequential data, like music, videos, text sequences or basically any sequence of
symbols. Without knowing what the symbols are or their meaning, they infer the underlying
semantics by analysing the sequence and the relative positions of the symbols.
They accept an input sequence x and give you an output sequence y. However, crucially this
output vector’s contents are influenced not only by the input you just fed in, but also on the
entire history of inputs you’ve fed in in the past. Thus it enables the RNN to hold a hidden state
that it gets to update every time step is called. This hidden state, or memory is the essence of
the input sequence in numeric form.
The RNNs used are Dynamic RNNs which allow for variable sequence lengths, ideal for a
chatbot setting.

Steps:

1)Run the pip3 install -r req.txt file first to install dependencies
2)Then run the training.py file to the train the deep learning model. 
3) Finally run the chatbot.py file 
and flask will run it on a local server 127.0.0.1:5000 .
