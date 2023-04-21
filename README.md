# Helpdesk-topics
Ciphix challenge to find the 10 most relevant topics in service desk messages.
Thanks for this interesting challenge!

The topic predictor is deployed on Heroku at:
https://ciphix-diederik.herokuapp.com/

It may take a while to load because it uses (free) transient resources.

## Model choice

For choosing the model I considered 3 main options:

-NMF
-LDA
-BERTopic

All of them would have been good choices. The main reason I preferred NMF and LDA is because
we do not have any labeled data and these are truly unsupervised models.
BERT would have worked well maybe, but relies in some form of supervision or 'guidance' woth seed words according to the Github page:
https://github.com/MaartenGr/BERTopic
Because I had no way to know what topics the helpdesk would be interested in for automation, I chose to opt out of BERTopic.

For the choice between NMF and LDA I preferred NMF because it works well on shorter messages and is commonly used for Twitter messages, while the helpdesk dataset looks similar to Twitter messages.

## Repo structure

The main branch was my working branch. The research and the created topics can be found in  `Topic_model.ipynb` in the Experiments folder.

To deploy the Django app to Heroku I needed it to have a specific folder structure, so it has a separate apps_only branch for that.
