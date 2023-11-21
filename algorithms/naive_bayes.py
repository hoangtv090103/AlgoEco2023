import pandas as pd


def main(sample_data=''):
    email = sample_data[0] if len(sample_data) > 0 else sample_data
    spam_dat = pd.read_csv('datasets/SMSSpamCollection', sep='\t', header=None, names=['Label', 'SMS'])
    spam_dat.head()
    print(spam_dat.Label.value_counts(normalize=True) * 100)
    print(spam_dat.shape)
    rand_spam = spam_dat.sample(frac=1, random_state=1)
    split_index = round(0.8 * len(rand_spam))

    train = rand_spam.iloc[:split_index, :].copy()
    test = rand_spam.iloc[split_index:, :].copy()
    train['SMS'] = train['SMS'].str.replace('\W', ' ')
    train['SMS'] = train['SMS'].str.lower()
    train.head()
    train['SMS'] = train['SMS'].str.split()

    vocabulary = []
    for sentence in train['SMS']:
        for word in sentence:
            vocabulary.append(word)
    len(vocabulary)
    vocabulary = list(set(vocabulary))
    len(vocabulary)
    word_counts_per_sms = {unique_word: [0] * len(train['SMS']) for unique_word in vocabulary}

    for index, sms in enumerate(train['SMS']):
        for word in sms:
            word_counts_per_sms[word][index] += 1

    word_counts = pd.DataFrame(word_counts_per_sms)
    word_counts.head()
    train_word = pd.concat([train, word_counts], axis=1)
    train_word.head()
    train_word = train_word[pd.notnull(train_word['Label'])]
    train_word.head()
    # Isolating spam and ham messages first
    spam_messages = train_word[train_word['Label'] == 'spam']
    ham_messages = train_word[train_word['Label'] == 'ham']

    # P(Spam) and P(Ham)
    p_spam = len(spam_messages) / len(train_word)
    p_ham = len(ham_messages) / len(train_word)

    # N_Spam
    n_words_per_spam_message = spam_messages['SMS'].apply(len)
    n_spam = n_words_per_spam_message.sum()

    # N_Ham
    n_words_per_ham_message = ham_messages['SMS'].apply(len)
    n_ham = n_words_per_ham_message.sum()

    # N_Vocabulary
    n_vocabulary = len(vocabulary)

    # Laplace smoothing
    alpha = 1
    ham_dict = {unique_words: 0 for unique_words in vocabulary}
    spam_dict = {unique_words: 0 for unique_words in vocabulary}

    for word in vocabulary:
        n_word_given_spam = spam_messages[word].sum()   # spam_messages already defined in a cell above
        p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
        spam_dict[word] = p_word_given_spam

        n_word_given_ham = ham_messages[word].sum()   # ham_messages already defined in a cell above
        p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
        ham_dict[word] = p_word_given_ham
    import re

    def classify(message):

        message = re.sub('\W', ' ', message)
        message = message.lower()
        message = message.split()

        p_spam_given_message = p_spam
        p_ham_given_message = p_ham

        for word in message:
            if word in spam_dict:
                p_spam_given_message *= spam_dict[word]

            if word in ham_dict:
                p_ham_given_message *= ham_dict[word]

        print('P(Spam|message):', p_spam_given_message)
        print('P(Ham|message):', p_ham_given_message)

        if p_ham_given_message > p_spam_given_message:
            return 'Label: Ham'
        elif p_ham_given_message < p_spam_given_message:
            return 'Label: Spam'
        else:
            return 'Equal proabilities, have a human classify this!'

    def classify_test_set(message):

        message = re.sub('\W', ' ', message)
        message = message.lower()
        message = message.split()

        p_spam_given_message = p_spam
        p_ham_given_message = p_ham

        for word in message:
            if word in spam_dict:
                p_spam_given_message *= spam_dict[word]

            if word in ham_dict:
                p_ham_given_message *= ham_dict[word]

        if p_ham_given_message > p_spam_given_message:
            return 'ham'
        elif p_spam_given_message > p_ham_given_message:
            return 'spam'
        else:
            return 'needs human classification'
    test['predicted'] = test['SMS'].apply(classify_test_set)
    test.head()

    return classify(email)


if __name__ == "_main_":
    main()
