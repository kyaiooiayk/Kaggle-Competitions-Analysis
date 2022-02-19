# Kaggle-Competitions-Analysis
First off, this is a work in progress repository. Each entry has 5 bullet points:
  - `Type`
  - `Competition overview`
  - `Winner blog`
  - `Winner notebook/code/kernel` 
  - `Other notebook/code/kernel`
  - `Solution thread`
  - `Take home message` (Taken directly from the authors with some modifications)
 
The list of competitions was taken from this [reference#1](https://farid.one/kaggle-solutions/) and [reference#2](https://www.kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions). Generally at the end of every Kaggle competition, the winners share their solutions. The goal of this repository is to offer a quick refence guide to the what matters the most: their kernel and the take home message. Competitions where neither the code nor the approach was described were ommited.

## A very short intro on Kaggle competition via bullet points
- Competitions on Kaggle are classified into different types according to their reward: Knowledge, Jobs, money. Knowledge competitions are meant for beginners who are looking to get started. These are a good fit for a beginner, because you can find a lot of articles and sample solutions explaining how to get a good score.
- After getting familiar with the platform and how to solve a competition, you can join a real live competition and participate.
- Kaggle has a rewarding system which categorise sers into Novice for recently joined users, Contributor, Expert, Master and GrandMaster for each of the four paradigms, Kaggle GrandMasters. The last being the highest rank achiavable,
- The Kaggle leaderboard has a public and private component to prevent participants from “overfitting” to the leaderboard. If your model is “overfit” to a dataset then it is not generalizable outside of the dataset you trained it on. This means that your model would have low accuracy on another sample of data taken from a similar dataset.
- Kaggle Kernels is a cloud computational environment that enables reproducible and collaborative analysis. Kernels supports scripts in R and Python, Jupyter Notebooks, and RMarkdown reports

## Introductory Article on Kaggle competition
- [How to get started with Kaggle](https://machinelearningmastery.com/get-started-with-kaggle/)
- [Quora, How do Kaggle competitions work?](https://www.quora.com/How-do-Kaggle-competitions-work)
- [FAQs by Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/frequently-asked-questions)

## Other resources on how to approach a Kaggle competition
- [How I Made It To The Kaggle Leaderboard](https://wandb.ai/lavanyashukla/kaggle-feature-encoding/reports/How-I-Made-It-To-The-Kaggle-Leaderboard--Vmlldzo2NzQyNQ)
- [How to (almost) win Kaggle competitions](https://yanirseroussi.com/2014/08/24/how-to-almost-win-kaggle-competitions/)
- [How to “farm” Kaggle in the right way](https://towardsdatascience.com/how-to-farm-kaggle-in-the-right-way-b27f781b78da)
- [What does it take to win a Kaggle competition? Let’s hear it from the winner himself](https://www.h2o.ai/blog/what-does-it-take-to-win-a-kaggle-competition-lets-hear-it-from-the-winner-himself/)
- [How We Placed in Kaggle Competition Top 4%](https://towardsdatascience.com/how-we-two-beginners-placed-in-kaggle-competition-top-4-3ea508638f2d)
- [Competing in a data science contest without reading the data](https://blog.mrtz.org/2015/03/09/competition.html)
- [Model Selection Tips From Competitive Machine Learning](https://machinelearningmastery.com/model-selection-tips-from-competitive-machine-learning/)
- [How to kick ass in competitive machine learning](https://machinelearningmastery.com/how-to-kick-ass-in-competitive-machine-learning/)
- [How to Select Your Final Models in a Kaggle Competition](http://www.chioka.in/how-to-select-your-final-models-in-a-kaggle-competitio/)

## Kaggle Grandmasters
- [Bo](https://www.kaggle.com/boliu0) | [GitHub](https://github.com/boliu61)
- [Chris Deotte](https://www.kaggle.com/cdeotte)
- [DungNB](https://www.kaggle.com/nguyenbadung) | [GitHub](https://github.com/dungnb1333)
- [Jean-François Puget](https://github.com/jfpuget) 
- [NguyenThanhNhan](https://www.kaggle.com/andy2709)
- [yelan](https://www.kaggle.com/lanjunyelan)
- [SeungKee](https://www.kaggle.com/keetar)
- [Tom Van de Wiele]() | [GitHub](https://github.com/ttvand)
- [Prashant Banerjee](https://www.kaggle.com/prashant111)
- [Firat Gonen](https://www.kaggle.com/frtgnn)
- [Laura Fink](https://www.kaggle.com/allunia)
- [Janio Martinez Bachmann](https://www.kaggle.com/janiobachmann/code?userId=1245336&sortBy=dateRun&tab=profile)

## Installing Kaggle API
- With pip: `pip install kaggle`
- [How to resolve kaggle.json not found](https://github.com/Kaggle/kaggle-api/issues/15).  Go to the Kaggle's homepage www.kaggle.com -> Your Account -> Create New API token. This will download a ready-to-go JSON file to place in you `[user-home]/.kaggle` folder. If there is no `.kaggle` folder yet, please create it first, however it is highly likely that the folder is already there, especially if you tried early this: `kaggle competitions download -c competition_name`.

## Kaggle API
You have two options to send over your submission: 1) directly from a Kaggle kernel or by their API. The last one is the one I prefer. I'd like to do all the wrangling and modelling on my set up and then send my submission file directly.
- [How to submit from kaggle kernel](https://www.kaggle.com/dansbecker/submitting-from-a-kernel)
- [Kaggle API wiki](https://github.com/Kaggle/kaggle-api)
- Step-by-step manual submission to Kaggle:
  - How to downlaod the sets? `kaggle competitions download -c house-prices-advanced-regression-techniques`
  - Where do I get the exact name of the competition? Check the URL like this: `https://www.kaggle.com/c/house-prices-advanced-regression-techniques/submissions`
  - See your submissions history: `kaggle competitions submissions house-prices-advanced-regression-techniques`
  - How to submit your file via Kaggle API: `kaggle competitions submit house-prices-advanced-regression-techniques -f submission.csv -m "Submission_No_1"`

## Notebale Techniques
- [Pesudo Labelling #1](https://www.kaggle.com/nroman/i-m-overfitting-and-i-know-it), [#2](https://www.kaggle.com/cdeotte/pseudo-labeling-qda-0-969)

## Useful resource
- [heamy](https://github.com/rushter/heamy) A set of useful tools for competitive data science. Automatic caching (data preprocessing, predictions from models) Ensemble learning (stacking, blending, weighted average, etc.).

## 1 Forecast Eurovision Voting
*This competition requires contestants to forecast the voting for this years Eurovision Song Contest in Norway on May 25th, 27th and 29th.*
  - [Competition overview](https://www.kaggle.com/c/Eurovision2010)
  - [Winner blog/article](https://medium.com/kaggle-blog/computer-scientist-jure-zbontar-on-winning-the-eurovision-challenge-8b5754fc72b4)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message: CV was paramount to avoid overfitting and provide an indication of which model would perform well on new data. It is stated that studying the voting patterns for all countries provided valuable insights.

## 2 Predict HIV Progression
*This contest requires competitors to predict the likelihood that an HIV patient's infection will become less severe, given a small dataset and limited clinical information.*
  - [Competition overview](https://www.kaggle.com/c/hivprogression/overview)
  - [Winner blog/article](https://medium.com/kaggle-blog/how-i-won-the-predict-hiv-progression-data-mining-competition-fbb7b682b7ef)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message: Make sure that all areas of the dataset are randomly partitioned. In order to do machine learning correctly, it is important to have your training data closely match the test dataset. Furtherm the recursive feature elimination was mentioned as one of the factor that helped win the competition.

## 3 Tourism Forecasting Part One
*Part one requires competitors to predict 518 tourism-related time series. The winner of this competition will be invited to contribute a discussion paper to the International Journal of Forecasting.*
  - [Competition overview](https://medium.com/kaggle-blog/how-i-did-it-lee-baker-on-winning-tourism-forecasting-part-one-3c3c9d1efcbc)
  - [Winner blog/article](https://medium.com/kaggle-blog/how-i-did-it-lee-baker-on-winning-tourism-forecasting-part-one-3c3c9d1efcbc)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message: A weighted combination of three predictors turned out to be the best appraoch for forecasting.

## 4 Tourism Forecasting Part Two
*Part two requires competitors to predict 793 tourism-related time series. The winner of this competition will be invited to contribute a discussion paper to the International Journal of Forecasting.*
  - [Competition overview](https://www.kaggle.com/c/tourism2)
  - [Winner blog/article](https://medium.com/kaggle-blog/phil-brierley-on-winning-tourism-forecasting-part-two-5aaa91b93e06)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message: The mindset was not to concentrate on the the overall accuracy, but how to prevent the worst case events. This was achieved an ensemble of algorithms.
  
## 5 INFORMS Data Mining Contest 2010
*The goal of this contest is to predict short term movements in stock prices. The winners of this contest will be honoured of the INFORMS Annual Meeting in Austin-Texas (November 7-10).*
  - [Competition overview](https://www.kaggle.com/c/informs2010/overview)
  - [Winner blog/article](https://medium.com/kaggle-blog/how-i-did-it-the-top-three-from-the-2010-informs-data-mining-contest-5267983308dd)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message: NA

## 6 Chess ratings - Elo versus the Rest of the World
*This competition aims to discover whether other approaches can predict the outcome of chess games more accurately than the workhorse Elo rating system.*
  - [Competition overview](https://www.kaggle.com/c/chess/overview)
  - [Winner blog/article](https://arxiv.org/abs/1012.4571)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message: The winning algorithm, called Elo++, is characterised by an l2 regularization technique that avoids overfitting. This was paramount given the extremey small dataset. Overfitting is a big problem for rating systems.  The regularization takes into account the number of games per player, the recency of these games and the ratings of the opponents of each player. The intuition is that any rating system should “trust” more the ratings of players who have played a lot of recent games versus the ratings of players who have played a few old games. The extent of regularization is controlled using a single parameter, that was optimised via CV. 
 
## 7 IJCNN Social Network Challenge
*This competition requires participants to predict edges in an online social network. The winner will receive free registration and the opportunity to present their solution at IJCNN 2011.*
  - [Competition overview](https://www.kaggle.com/c/socialNetwork/overview)
  - [Winner blog/article](https://arxiv.org/abs/1102.4374)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message: Apart from the techniality of the winning approach, the most interesting finding was that large real-world online social network graph can be effectively de-anonymised. Releasing useful social network graph data that is resilient to de-anonymisation remains an open question.
 
## 8 R Package Recommendation Engine
*The aim of this competition is to develop a recommendation engine for R libraries (or packages). (R is opensource statistics software.*
  - [Competition overview](https://www.kaggle.com/c/R/overview)
  - [Winner (2nd) blog/article](https://medium.com/kaggle-blog/max-lin-on-finishing-second-in-the-r-challenge-520a7d785beb)
  - [Winner (2nd) notebook/code/kernel](https://github.com/m4xl1n/r_recommendation_system)
  - Other notebook/code/kernel - NA
  - Take home message: ensamble of 4 different models.
  
## 9 RTA Freeway Travel Time Prediction
*This competition requires participants to predict travel time on Sydney's M4 freeway from past travel time observations.*
  - [Competition overview](https://www.kaggle.com/c/RTA)
  - [Winner blog/article](https://1library.net/document/zp78l9rz-using-ensemble-decision-trees-forecast-travel-time.html)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message: NA

## 10 Predict Grant Applications
*This task requires participants to predict the outcome of grant applications for the University of Melbourne.*
  - [Competition overview](https://www.kaggle.com/c/unimelb)
  - [Winner blog/article](https://medium.com/kaggle-blog/jeremy-howard-on-winning-the-predict-grant-applications-competition-e70a252946c9)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message: Pre-processing of the Excel spreadsheet looking for groups which had high or low application success rates. The winning algorithm was a slight modification of the random forest algorithm.
  
## 11 Stay Alert! The Ford Challenge
*Driving while not alert can be deadly. The objective is to design a classifier that will detect whether the driver is alert or not alert, employing data that are acquired while driving.*
  - [Competition overview](https://www.kaggle.com/c/stayalert)
  - [Winner blog/article](https://web.archive.org/web/20160109015217/http://blog.kaggle.com/wp-content/uploads/2011/03/ford.pdf)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message: Trials are not homogeneous, meaning the driver is either mainly alert or not alert. High AUC on the training set (or a held-back portion of the training set) and achieves a poor AUC on the test set. This observation suggests that we are working in the world of extrapolation: i.e. the training and test set differ in some manner. If we’re extrapolating then a simple model is usually required. The winning algorithm was based on a simple logistic regression.

## 12 ICDAR 2011 - Arabic Writer Identification
*This competition require participants to develop an algorithm to identify who wrote which documents. The winner will be honored at a special session of the ICDAR 2011 conference.*
  - [Competition overview](https://www.kaggle.com/c/WIC2011/code)
  - [Winner (3rd) blog/article](https://medium.com/kaggle-blog/on-diffusion-kernels-histograms-and-arabic-writer-identification-f8b61a6afce8)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message: The approach was based on SVMs with a diffusion kernel. A notable comment was that there was an apparent lack of correlation between CV results on the training data and the accuracy on the validation set.
  
## 13 Deloitte/FIDE Chess Rating Challenge
*This contest, sponsored by professional services firm Deloitte, will find the most accurate system to predict chess outcomes, and FIDE will also bring a top finisher to Athens to present their system.*
  - [Competition overview](https://www.kaggle.com/c/ChessRatings2/leaderboard)
  - [Winner blog/article](https://medium.com/kaggle-blog/the-thrill-of-the-chase-tim-salimans-on-how-he-took-home-deloitte-fide-chess-comp-c4ee7b29b735)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message: Yannis Sismanis, the winner of the first competition, used a logistic curve for this purpose and estimated the rating numbers by minimizing a regularized version of the model fit. Jeremy Howard, the runner-up, instead used the TrueSkill model, which uses a Gaussian cumulative density function and estimates the ratings using approximate Bayesian inference. These were the starting points but the winning algorithm was based on some solid post-processing of the data. These inludes:
- the predictions of the base model
- the ratings of the players
- the number of matches played by each player
- the ratings of the opponents encountered by each player
- the variation in the quality of the opponents encountered
- the average predicted win percentage over all matches in the same month for each player
- the predictions of a random forest using these variables

## 14 Don't Overfit!
*With nearly as many variables as training cases, what are the best techniques to avoid disaster?*
  - [Competition overview](https://www.kaggle.com/c/overfitting)
  - [Winner blog/article](https://www.kaggle.com/c/overfitting/discussion/593)
  - [Winner (2nd) notebook/code/kernel](http://people.few.eur.nl/salimans/dontoverfit.html)
  - [Other notebook/code/kernel](https://medium.com/analytics-vidhya/kaggle-competition-dont-overfit-ii-74cf2d9deed5)
  - Take home message: 

## 15 Mapping Dark Matter
*Measure the small distortion in galaxy images caused by dark matter*
  - [Competition overview](https://www.kaggle.com/c/mdm/data)
  - [Winner blog/article](https://medium.com/kaggle-blog/deepzot-on-dark-matter-how-we-won-the-mapping-dark-matter-challenge-c3c7c09d9300)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message: Without the neural network, the winner best entry would have ranked 8th.

## 16 Wikipedia's Participation Challenge
*This competition challenges data-mining experts to build a predictive model that predicts the number of edits an editor will make five months from the end date of the training dataset*
  - [Competition overview](https://www.kaggle.com/c/wikichallenge)
  - [Winner (3rd) blog/article](https://medium.com/kaggle-blog/long-live-wikipedia-dell-zhang-on-placing-third-in-wikipedias-participation-challenge-c52be03ee257)
  - Other notebook/code/kernel - NA
  - Take home message: NA

## 17 dunnhumby's Shopper Challenge
*Going grocery shopping, we all have to do it, some even enjoy it, but can you predict it? dunnhumby is looking to build a model to better predict when supermarket shoppers will next visit the store and how much they will spend.*
  - [Competition overview](https://www.kaggle.com/c/dunnhumbychallenge)
  - [Winner blog/article](https://medium.com/kaggle-blog/kernel-density-at-the-checkout-dyakonov-alexander-on-winning-the-dunnhumby-shopper-challenge-b7103d108e3e)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message: At first I tried to use simple heuristics to understand the ‘logic of the problem’. My main idea was to split the problem into two parts: the date prediction and the dollar spend prediction. For that task, I used a kernel density (Parzen) estimator. But it was necessary to take account of the fact that ‘fresh’ data is more useful than ‘old’ data, so I used a weighted Parzen scheme to give greater weight to more recent data points.
  
## 18 Photo Quality Prediction
*Given anonymized information on thousands of photo albums, predict whether a human evaluator would mark them as 'good'.*
  - [Competition overview](https://www.kaggle.com/c/PhotoQualityPrediction/overview)
  - [Winner blog/article](https://medium.com/kaggle-blog/picture-perfect-bo-yang-on-winning-the-photo-quality-prediction-competition-f43c75efa7d6)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message: My best result was a mix of random forest, GBM, and two forms of logistic regression. I put the raw data into a database and built many derived variables. It’s probably not worth it to spend too much time on external data, as chances are any especially useful data are already included. Time can be better spent on algorithms and included variables. 
  
 ## 19 Give Me Some Credit
*Improve on the state of the art in credit scoring by predicting the probability that somebody will experience financial distress in the next two years. *
  - [Competition overview](https://www.kaggle.com/c/GiveMeSomeCredit)
  - [Winner blog/article](https://medium.com/kaggle-blog/the-perfect-storm-meet-the-winners-of-give-me-some-credit-97bcb4192f33)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message: We tried many different supervised learning methods, but we decided to keep our ensemble to only those things that we knew would improve our score through cross-validation evaluations. In the end we only used five supervised learning methods: a random forest of classification trees, a random forest of regression trees, a classification tree boosting algorithm, a regression tree boosting algorithm, and a neural network. This competition had a fairly simple data set and relatively few features which meant that the barrier to entry was low, competition would be very intense and everyone would eventually arrive at similar results and methods. Thus, I would have to work extra hard and be really innovative in my approach to solving this problem. I was surprised at how well neural networks performed. They certainly gave a good improvement over and above more modern approaches based on bagging and boosting. I have tried neural networks in other competitions where they did not perform as well.
  
 ## 20 Don't Get Kicked!
*Predict if a car purchased at auction is a lemon*
  - [Competition overview](https://www.kaggle.com/c/DontGetKicked)
  - [Winner (2nd) blog/article](https://medium.com/kaggle-blog/vladimir-nikulin-on-taking-2nd-prize-in-dont-get-kicked-79aafb91f9b8)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message: On the pre-processing: it was necessary to transfer textual values to the numerical format. I used Perl to do that task. Also, I created secondary synthetic variables by comparing different Prices/Costs. On the supervised learning methods: Neural Nets (CLOP, Matlab) and GBM in R. No other classifiers were user in order to produce my best result. Note that the NNs were used only for the calculation of the weighting coefficient in the blending model. Blending itself was conducted not around the different classifiers, but around the different training datasets with the same classifier. I derived this idea during last few days of the Contest, and it produced very good improvement (in both public and private).

 ## 21 Algorithmic Trading Challenge
*Develop new models to accurately predict the market response to large trades.
  - [Competition overview](https://www.kaggle.com/c/AlgorithmicTradingChallenge)
  - [Winner blog/article](https://medium.com/kaggle-blog/meet-the-winner-of-the-algo-trading-challenge-an-interview-with-ildefons-magrans-417d6a68c271)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - [Solution thread](https://www.kaggle.com/c/AlgorithmicTradingChallenge/discussion/1236)
  - Take home message: I tried many techniques: (SVM, LR, GBM, RF). Finally, I chose to use a random forest. The training set was a nice example of how stock market conditions are extremely volatile. Different samples of the training set could fit very different models.

## 22 The Hewlett Foundation: Automated Essay Scoring
*Develop an automated scoring algorithm for student-written essays.*
  - [Competition overview](https://www.kaggle.com/c/asap-aes)
  - Winner blog - NA
  - Winner notebook/code/kernel - NA
  - [Other notebook/code/kernel #1](https://www.kaggle.com/soumya9977/autograding-using-lstm-tf-keras)
  - [Other notebook/code/kernel #2](https://github.com/ogozuacik/automated_essay_scoring)
  - Take home message - NA

## 23 KDD Cup 2012, Track 2
*Predict the click-through rate of ads given the query and user information.*
  - [Competition overview](https://www.kaggle.com/c/kddcup2012-track2)
  - Winner blog - NA
  - Winner notebook/code/kernel - NA
  - [Other notebook/code/kernel #1](https://www.kaggle.com/shivashi11/ad-click-prediction)  
  - Take home message
 
## 24 Predicting a Biological Response
*Predict a biological response of molecules from their chemical properties.*
  - [Competition overview](https://www.kaggle.com/c/bioresponse/code)
  - Winner blog - NA
  - Winner notebook/code/kernel - NA
  - [Other notebook/code/kernel #1](https://www.kaggle.com/vernondsouza123/predict-biological-response-through-lightgbm)
  - Take home message

## 25 Facebook Recruiting Competition
*Show them your talent, not just your resume.*
  - [Competition overview](https://www.kaggle.com/c/FacebookRecruiting/code)
  - Winner blog - NA
  - Winner notebook/code/kernel - NA
  - [Other notebook/code/kernel #1](https://www.kaggle.com/genialgokul1099/social-network-graph-link-prediction)
  - Take home message
  
## 26 EMI Music Data Science Hackathon - July 21st - 24 hours
*Can you predict if a listener will love a new song?*
  - [Competition overview](https://www.kaggle.com/c/MusicHackathon/overview)
  - [Winner blog/article](https://github.com/fancyspeed/codes_of_innovations_for_emi/blob/master/technical_report_innovations.pdf)
  - [Winner notebook/code/kernel] (https://github.com/fancyspeed/codes_of_innovations_for_emi)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
 ## 27 Detecting Insults in Social Commentary
*Predict whether a comment posted during a public discussion is considered insulting to one of the participants.*
  - [Competition overview](https://www.kaggle.com/c/detecting-insults-in-social-commentary)
  - [Winner blog/article](https://www.kaggle.com/c/detecting-insults-in-social-commentary/discussion/2744#15951)
  - Winner notebook/code/kernel - NA
  - [Other notebook/code/kernel](https://www.kaggle.com/rishabhgarg1023/detecting-insults-from-social-commentary)
  - Take home message - NA

 ## 28 Predict Closed Questions on Stack Overflow
*Predict which new questions asked on Stack Overflow will be closed*
  - [Competition overview](https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow/overview)
  - [Winner blog/article]()
  - Winner notebook/code/kernel - NA
  - [Other notebook/code/kernel](https://github.com/l0gr1thm1k/Predict-Closed-Questions-on-Stack-Overflow)
  - Take home message - NA
  
 ## 29 Observing Dark Worlds
*Can you find the Dark Matter that dominates our Universe? Winton Capital offers you the chance to unlock the secrets of dark worlds.*
  - [Competition overview](https://www.kaggle.com/c/DarkWorlds/overview)
  - [Winner blog/article](https://jikeme.com/1st-place-observing-dark-worlds)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - Although the description makes this sound like a physics problem, it is really one of statistics: given the noisy data (the elliptical galaxies) recover the model and parameters (position and mass of the dark matter) that generated them. Bayesian analysis provided the winning recipe for solving this problem. The 1.05 public score of my winning submission was only about average on the public leaderboard. All of this means <u>I was very lucky<u> indeed to win this competition. 

 ## 30 Traveling Santa Problem
*Solve ye olde traveling salesman problem to help Santa Claus deliver his presents*
  - [Competition overview](https://www.kaggle.com/c/traveling-santa-problem/overview)
  - Winner blog - NA
  - Winner notebook/code/kernel - NA
  - [Other notebook/code/kernel](https://www.kaggle.com/javiabellan/starting-kernel-plotting-nearest-neighbor)
  - Take home message -NA
   
 ## 31 Event Recommendation Engine Challenge
*Predict what events our users will be interested in based on user actions, event metadata, and demographic information.*
  - [Competition overview](https://www.kaggle.com/c/event-recommendation-engine-challenge/overview)
  - Winner blog - NA
  - Winner notebook/code/kernel - NA
  - [Other notebook/code/kernel](https://www.kaggle.com/tombalu/event-recommendation-mldm-eda)
  - Take home message -NA
  
## 32 Job Salary Prediction
*Predict the salary of any UK job ad based on its contents.*
  - [Competition overview](https://www.kaggle.com/c/job-salary-prediction/overview)
  - Winner blog - NA
  - Winner notebook/code/kernel - NA
  - [Other notebook/code/kernel](https://www.kaggle.com/jillanisofttech/job-salary-prediction-by-jst)
  - Take home message - NA

## 33 Influencers in Social Networks
*Predict which people are influential in a social network.*
  - [Competition overview](https://www.kaggle.com/c/predict-who-is-more-influential-in-a-social-network/code)
  - Winner blog - NA
  - Winner notebook/code/kernel - NA
  - [Other notebook/code/kernel](https://www.kaggle.com/khotijahs1/influencers-in-social-networks)
  - Take home message - NA

## 34 Blue Book for Bulldozers
*Predict the auction sale price for a piece of heavy equipment to create a "blue book" for bulldozers.*
  - [Competition overview](https://www.kaggle.com/c/bluebook-for-bulldozers/overview)
  - [Winner (20th place) blog/article](https://blog.dataiku.com/2013/04/26/kaggle-contest-blue-book-for-bulldozers)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - Kaggle provided us with a machine appendix with the “real” value of each feature and for each machine, but it turned out that putting in the true value **was not a good idea**. Indeed, we think that each seller could declare the characteristics (or not) on the auction website and this had an impact on the price. As for the second point, we focused on the volatility of some models. We spent a lot of time trying to understand how a machine could be sold the same year, and, even with only a few days between two sales, at two completely different prices. It turned out not to be easily predictable. In financial theory, the model used to describe this kind of randomness is call **random walk**. We tried a lot of things: we decomposed each option in new binary features, we added the age from the sale date and the year of manufacture, we added the day of the week, the number of the week in the year, we also tried to add the number of auctions of the current month to try to capture the market tendency, we tried to learn our models on different periods, for example by removing the year 2009 and 2010 which were impacted by the economic crisis. In the end we built **one model per category.**

## 35 Challenges in Representation Learning: Multi-modal Learning
*The multi-modal learning challenge.*
  - [Competition overview](https://www.kaggle.com/c/challenges-in-representation-learning-multi-modal-learning/overview)
  - [Winner (20th place) blog/article](https://www.kaggle.com/c/challenges-in-representation-learning-multi-modal-learning/discussion/4727)
  - [Winner notebook/code/kernel](https://code.google.com/archive/p/image-txt-multimodal/)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 36 Challenges in Representation Learning: The Black Box Learning Challenge
*Competitors train a classifier on a dataset that is not human readable, without knowledge of what the data consists of.*
  - [Competition overview](https://www.kaggle.com/c/challenges-in-representation-learning-the-black-box-learning-challenge/overview)
  - [Winner blog/article](https://www.kaggle.com/c/challenges-in-representation-learning-the-black-box-learning-challenge/discussion/4717)
  - [Winner notebook/code/kernel/code](https://bitbucket.org/dthal/blackbox-challenge-code/src/master/)
  - Other notebook/code/kernel - NA
  - Take home message - Although almost all of the winner submissions were single classifiers, the actual winning entry was a small ensemble of three previous submissions.
  
## 37 Challenges in Representation Learning: Facial Expression Recognition Challenge
*Learn facial expressions from an image*
  - [Competition overview]()
  - [Winner blog/article](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/discussion/4676#25019)
  - [Winner notebook/code/kernel](https://code.google.com/archive/p/deep-learning-faces/)
  - Other notebook/code/kernel - NA
  - Take home message - NA
 
## 38 KDD Cup 2013 - Author Disambiguation Challenge (Track 2)
*Identify which authors correspond to the same person*
  - [Competition overview](https://www.kaggle.com/c/kdd-cup-2013-author-disambiguation)
  - [Winner blog/article](https://www.csie.ntu.edu.tw/%7Ecjlin/papers/kddcup2013/)
  - [Winner (2nd) notebook/code/kernel](https://github.com/remenberl/KDDCup2013)
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 39  The ICML 2013 Whale Challenge - Right Whale Redux
*Develop recognition solutions to detect and classify right whales for BIG data mining and exploration studies*
  - [Competition overview](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/overview)
  - [Winner blog/article]()
  - [Winner (2nd) notebook/code/kernel](https://github.com/felixlaumon/kaggle-right-whale)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 40 KDD Cup 2013 - Author-Paper Identification Challenge (Track 1)
*Determine whether an author has written a given paper*
  - [Competition overview](https://www.kaggle.com/c/kdd-cup-2013-author-paper-identification-challenge)
  - [Winner blog/article](https://www.csie.ntu.edu.tw/%7Ecjlin/papers/kddcup2013/)
  - [Winner notebook/code/kernel]()
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 41 Amazon.com - Employee Access Challenge
*Predict an employee's access needs, given his/her job role*
  - [Competition overview](https://www.kaggle.com/c/amazon-employee-access-challenge)
  - [Winner blog/article](https://github.com/bensolucky/Amazon)
  - [Winner notebook/code/kernel](https://github.com/bensolucky/Amazon)
  - [Other notebook/code/kernel] #1(https://github.com/kaz-Anova/Competitive_Dai)
  - [Other notebook/code/kernel] #1(https://github.com/kaz-Anova/ensemble_amazon)
  - Take home message  - The general strategy was to produce 2 feature sets: one categorical to be modeled with decision tree based approaches and the second a sparse matrix of binary features, created by binarizing all categorical values and 2nd and 3rd order combinations of categorical values. The latter features could be modeled with Logistic Regressoin, SVMs, etc. The starting point of this latter set of code was provided on the forums by Miroslaw Horbal. The most critical modeification I made to it was in merging the most rarely occuring binary features into a much smaller number of features holding these rare values. 
 
## 42 MLSP 2013 Bird Classification Challenge
*Predict the set of bird species present in an audio recording, collected in field conditions.*
  - [Competition overview](https://www.kaggle.com/c/mlsp-2013-birds)
  - [Winner blog/article](https://www.kaggle.com/c/mlsp-2013-birds/discussion/5457#29159)
  - [Winner notebook/code/kernel](https://github.com/gaborfodor/MLSP_2013)
  - Other notebook/code/kernel - NA
  - Take home message - NA   

## 43 RecSys2013: Yelp Business Rating Prediction
*RecSys Challenge 2013: Yelp business rating prediction*
  - [Competition overview](https://www.kaggle.com/c/yelp-recsys-2013)
  - [Winner blog/article]()
  - [Winner (7th) notebook/code/kernel](https://github.com/theusual/kaggle-yelp-business-rating-prediction)
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 44 The Big Data Combine Engineered by BattleFin
*Predict short term movements in stock prices using news and sentiment data provided by RavenPack*
  - [Competition overview](https://www.kaggle.com/c/battlefin-s-big-data-combine-forecasting-challenge/overview)
  - [Winner blog/article](https://www.kaggle.com/c/battlefin-s-big-data-combine-forecasting-challenge/discussion/5941)
  - [Winner notebook/code/kernel]()
  - Other notebook/code/kernel - NA
  - Take home message - A high-level description of my approach is: 1. Group securities into groups according to price movement correlation. 2. For each security group, use I146 to build a “decision stump” (a 1-split decision tree with 2 leaf nodes). 3. For each leaf node, build a model of the form Prediction = m * Last Observed Value. For each leaf node, find m that minimizes MAE. Rows that most-improved or most-hurt MAE with respect to m=1.0 were not included. 

## 45 Belkin Energy Disaggregation Competition
*Disaggregate household energy consumption into individual appliances*
  - [Competition overview]()
  - [Winner (discussion) blog/article](https://www.kaggle.com/c/belkin-energy-disaggregation-competition/discussion/6168)
  - [Winner notebook/code/kernel]()
  - Other notebook/code/kernel - NA
  - Take home message - The winner of the competition shown by the final standings was actually only ranked 6th on the public leaderboard. This suggests that many participants might have been **overfitting** their algorithms to the half of the test data for which the performance was disclosed, while the competition winner had not optimised their approach in such a way. An interesting forum thread seems to show that most successful participants used an approach based on only low-frequency data, despite the fact that high-frequency data was also provided. This seems to contradict most academic research, which generally shows that high-frequency based approaches will outperform low-frequency methods. A reason for this could be that, although high-frequency based approaches perform well in laboratory test environments, their features do not generalise well over time, and as a result algorithm training quickly becomes outdated. However, another reason could have been that the processing of the high-frequency features was simply too time consuming, and better performance could be achieved by concentrating on the low-frequency data given the deadline of the competition.

## 46 StumbleUpon Evergreen Classification Challenge
*Build a classifier to categorize webpages as evergreen or non-evergreen*
  - [Competition overview](https://www.kaggle.com/c/stumbleupon)
  - [Winner blog/article]()
  - [Winner (2nd) notebook/code/kernel](https://github.com/ma2rten/kaggle-evergreen)
  - [Other (4th) notebook/code/kernel](https://github.com/saffsd/kaggle-stumbleupon2013)
  - Take home message - NA 

## 47  AMS 2013-2014 Solar Energy Prediction Contest
*Forecast daily solar energy with an ensemble of weather models*
  - [Competition overview](https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest/leaderboard)
  - [Winner blog/article](https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest/discussion/6321)
  - [Winner notebook/code/kernel](https://github.com/lucaseustaquio/ams-2013-2014-solar-energy)
  - [Other notebook/code/kernel]
  - Take home message - NA 
 
## 48 Accelerometer Biometric Competition
*Recognize users of mobile devices from accelerometer data*
  - [Competition overview](https://www.kaggle.com/c/accelerometer-biometric-competition)
  - [Winner (1st, 2nd, 3d, 4th) blog/article](https://www.kaggle.com/c/accelerometer-biometric-competition/discussion/6474)
  - [Winner (3rd) notebook/code/kernel](https://github.com/rouli/kaggle_accelerometer)
  - Other notebook/code/kernel - NA
  - Take home message - Main idea is constructing chains of consecutive sequences using timestamp
leak and determining real device using bayes rule and professed devices of se-
quences in this chain. Some chains are “stuck” on their real devices 
  
## 49 Multi-label Bird Species Classification - NIPS 2013
*Identify which of 87 classes of birds and amphibians are present into 1000 continuous wild sound recordings*
  - [Competition overview](https://www.kaggle.com/c/multilabel-bird-species-classification-nips2013)
  - [Winner blog/article]()
  - [Winner (2nd) notebook/code/kernel](https://github.com/mattwescott/bird-recognition)
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 50 See Click Predict Fix
*Predict which 311 issues are most important to citizens*
  - [Competition overview](https://www.kaggle.com/c/see-click-predict-fix)
  - [Winner (2nd) blog/article](https://www.kaggle.com/c/see-click-predict-fix/discussion/6754)
  - [Winner notebook/code/kernel](https://github.com/BlindApe/SeeClickPredictFix)
  - Other notebook/code/kernel - NA
  - Take home message - Because this contest was temporal in nature, using time-series models to make future predictions, most competitors quickly realized that proper calibration of predictions was a major factor in reducing error.  Even during the initial Hackathon portion of the contest, it became well known on the competition forum that one needed to apply scalars to predictions in order to optimize leaderboard scores. But while scaling was common knowledge, our most important insight came in applying our segmentation approach to the scalars.  For example, rather than apply one optimized scalar to all predicted views for the entire test set, we applied optimized scalars for each distinct segment of the test set (the remote API sourced issues and the four cities).  We then optimized the scalars using a combination of leaderboard feedback and cross-validation scores.  What we found was that each segment responded differently to scaling, so trying to apply one scalar to all issues, as many of our competitors were doing, was not optimal.  
  
## 51 Partly Sunny with a Chance of Hashtags
*What can a #machine learn from tweets about the #weather?*
  - [Competition overview](https://www.kaggle.com/c/crowdflower-weather-twitter)
  - [Winner blog/article](https://www.kaggle.com/c/crowdflower-weather-twitter/discussion/6488)
  - Winner (2nd) notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - Regarding the ML model, one core observation, which I guess prevented many people from entering into the <.15 zone, is that the problem here is multi-output. While the Ridge does handle the multi-output case it actually treats each variable independently. You could easily verify this by training an individual model for each of the variables and compare the results. You would see the same performance. So, the core idea is how to go about taking into account the correlations between the output variables. The approach I took was simple stacking, where you feed the output of a first level model and use it as features to the 2nd level model (of course you do it in a CV fashion).

## 52 Facebook Recruiting III - Keyword Extraction
*Identify keywords and tags from millions of text questions*
  - [Competition overview](https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction)
  - [Winner blog/article](https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/discussion/6650)
  - [Other relevant blog/article](https://www.analyticsvidhya.com/blog/2015/07/top-10-kaggle-fb-recruiting-competition/)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - As noted by everyone, the bottleneck of this competition is **RAM**.
  
## 53 Personalized Web Search Challenge
*Re-rank web documents using personal preferences*
  - [Competition overview](https://www.kaggle.com/c/yandex-personalized-web-search-challenge/overview)
  - Winner blog
  - Winner notebook/code/kernel - NA
  - [Other notebook/code/kernel] (https://github.com/ykdojo/personalized_search_challenge)
  - Take home message - NA   

## 54 Packing Santa's Sleigh
*He's making a list, checking it twice; to fill up his sleigh, he needs your advice*
  - [Competition overview](https://www.kaggle.com/c/packing-santas-sleigh/overview)
  - [Winners blog/article](https://www.kaggle.com/c/packing-santas-sleigh/discussion/6934#post38414)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA   
  
## 55 Dogs vs. Cats
*Create an algorithm to distinguish dogs from cats*
  - [Competition overview](https://www.kaggle.com/c/dogs-vs-cats/overview)
  - [Winner (4th) blog/article](https://medium.com/kaggle-blog/dogs-vs-cats-redux-playground-competition-winners-interview-bojan-tunguz-7233c12e03bf)
  - Winner  notebook/code/kernel -NA
  - Other notebook/code/kernel - NA
  - Take home message - Just like with most other image recognition/classification problems, I have completely relied on Deep Convolutional Neural Networks (DCNN). I have built a simple convolutional neural network (CNN) in Keras from scratch, but for the most part I’ve relied on out-of-the-box models: VGG16, VGG19, Inception V3, Xception, and various flavors of ResNets. My simple CNN managed to get the score in the 0.2x range on the public leaderboard (PL). My best models that I build using features extracted by applying retrained DCNNs got me into the 0.06x range on PL. Stacking of those models got me in the 0.05x range on PL. My single best fine-tuned DCNN got me to 0.042 on PL, and my final ensemble gave me the 0.35 score on PL. My ensembling diagram can be seen below: 

## 56 Conway's Reverse Game of Life
*Reverse the arrow of time in the Game of Life*
  - [Competition overview](https://www.kaggle.com/c/conway-s-reverse-game-of-life)
  - [Winner blog/article](https://www.kaggle.com/c/conway-s-reverse-game-of-life/discussion/7254)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - Regarding better hardware, I had figured that some might have access to more computing power, but since probability convergence has exponentially decreasing returns over time, I felt I could do just fine provided I used the information as well as possible. I tuned the ~100 parameters using genetic algorithms, mostly because I was too lazy to come up with something more theoretically rigorous in terms of optimizing capabilitie. 
   
## 57 Loan Default Prediction - Imperial College London
*Constructing an optimal portfolio of loans*
  - [Competition overview](https://www.kaggle.com/c/loan-default-prediction)
  - [Winner blog/article]()
  - [Winner (2nd) notebook/code/kernel](https://github.com/freedomljc/Loan_Default_Prediction)
  - [Other (12th) notebook/code/kernel](https://github.com/dmcgarry/Default_Loan_Prediction)
  - Take home message - The training data is sorted by the time, and the test data is randomly orded. So in the validation process, I first shuffle the training data randomly. **Owing to lack of the feature description**, It is hard to use the tradition method to predict LGD. In my implemention, the operator +,-.*,/  between two features, and the operator (a-b) * c among three features were used, these features were selected by computing the pearson corrlation with the loss.
   
## 58 Galaxy Zoo - The Galaxy Challenge
*Classify the morphologies of distant galaxies in our Universe*
  - [Competition overview](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge)
  - [Winner blog/article](https://benanne.github.io/2014/04/05/galaxy-zoo.html)
  - [Winner notebook/code/kernel](https://github.com/benanne/kaggle-galaxies)
  - Other notebook/code/kernel - NA
  - Take home message - NA  
 
## 59 March Machine Learning Mania
*Tip off college basketball by predicting the 2014 NCAA Tournament*
  - [Competition overview](https://www.kaggle.com/c/march-machine-learning-mania-2014/overview)
  - [Winner blog/article](https://medium.com/kaggle-blog/march-machine-learning-mania-1st-place-winners-interview-andrew-landgraf-f18214efc659)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - Sometimes it’s better to be lucky than good. The location data that I used had a coding error in it. South Carolina’s Sweet Sixteen and Elite Eight games were coded as being in Greenville, SC instead of New York City. The led me to give them higher odds than most others, which helped me since they won. It is hard to say what the optimizer would have selected (and how it affected others’ models), but there is a good chance I would have finished in 2nd place or worse if the correct locations had been used.  
  
## 60 Large Scale Hierarchical Text Classification
*Classify Wikipedia documents into one of 325,056 categories*
  - [Competition overview](https://www.kaggle.com/c/lshtc)
  - [Winner blog/article](https://www.kaggle.com/c/lshtc/discussion/7980)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - Our winning submission consists mostly of an ensemble of sparse generative models extending Multinomial Naive Bayes. The base-classifiers consist of hierarchically smoothed models combining document, label, and hierarchy level Multinomials, with feature pre-processing using variants of TF-IDF and BM25. Additional diversification is introduced by different types of folds and random search optimization for different measures. The ensemble algorithm optimizes macroFscore by predicting the documents for each label, instead of the usual prediction of labels per document. Scores for documents are predicted by weighted voting of base-classifier outputs with a variant of Feature-Weighted Linear Stacking. The number of documents per label is chosen using label priors and thresholding of vote scores.  
   
## 61 Walmart Recruiting - Store Sales Forecasting
*Use historical markdown data to predict store sales*
  - [Competition overview](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
  - [Winner (2nd) blog/article](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8023)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - I used SAS (for data prep/ARIMA/UCM) and R (for the remainder models) together. I used weighted average and trimmed mean of following  6 methods. The goal from  the beginning was to build a robust model that will be able to withstand uncertainty.
   
## 62 The Analytics Edge (15.071x)
*Learn what predicts happiness by using informal polling questions.*
  - [Competition overview](https://www.kaggle.com/c/the-analytics-edge-mit-15-071x)
  - [Winner blog/article](https://www.kaggle.com/c/allstate-purchase-prediction-challenge/discussion/8218)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA 
   
## 63 CONNECTOMICS
*Reconstruct the wiring between neurons from fluorescence imaging of neural activity*
  - [Competition overview](https://www.kaggle.com/c/connectomics/overview)
  - [Winner blog/article](https://arxiv.org/pdf/1406.7865.pdf)
  - [Winner notebook/code/kernel](https://github.com/asutera/kaggle-connectomics)
  - Other notebook/code/kernel - NA
  - Take home message - NA 
   
## 64 Allstate Purchase Prediction Challenge
*Predict a purchased policy based on transaction history*
  - [Competition overview](https://www.kaggle.com/c/allstate-purchase-prediction-challenge)
  - [Winner blog/article]()
  - [Winner (2nd) notebook/code/kernel](https://github.com/alzmcr/allstate)
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 65 Greek Media Monitoring Multilabel Classification (WISE 2014)
*Multi-label classification of printed media articles to topics*
  - [Competition overview](https://www.kaggle.com/c/wise-2014)
  - [Winner blog/article](http://alexanderdyakonov.narod.ru/wise2014-kaggle-Dyakonov.pdf )
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 66 KDD Cup 2014 - Predicting Excitement at DonorsChoose.org
*Predict funding requests that deserve an A+*
  - [Competition overview](https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/overview)
  - [Winner blog/article](https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/discussion/9774)
  - [Winner notebook/code/kernel](https://github.com/yoonkim/kdd_2014)
  - Other notebook/code/kernel - NA
  - Take home message - We worked independently until we merged so we had two separate "models" (each "model" was itself an ensemble of a few models). A lot of our variables ended up being similar, though. Instead of listing out features one-by-one we will note some potentially interesting features that we used. The full feature list can be found in the code. Given private LB's sensitivity to discounting, and given public LB's (relative) lack of sensitivity to discounting (e.g. 1.0 to 0.5 linear decay gave ~0.003 improvements on the public LB), we were simply **lucky**. 

## 67 MLSP 2014 Schizophrenia Classification Challenge
*Diagnose schizophrenia using multimodal features from MRI scans*
  - [Competition overview](https://www.kaggle.com/c/mlsp-2014-mri)
  - [Winner blog/article](https://www.kaggle.com/c/mlsp-2014-mri/discussion/9907)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - The solution is based on Gaussian process classification. The model is actually very simple (really a 'Solution draft'), but it did show promising performance using LOOCV. The score on the public leaderboard was, however, only 0.70536, **discouraging** any further tuning of the model. In the end, it turned out to **perform well** on the private leaderboard.
  
## 68  DecMeg2014 - Decoding the Human Brain
*Predict visual stimuli from MEG recordings of human brain activity*
  - [Competition overview](https://www.kaggle.com/c/decoding-the-human-brain)
  - [Winner blog/article](https://github.com/alexandrebarachant/DecMeg2014/blob/master/doc/documentation.pdf)
  - [Winner (2nd) notebook/code/kernel](https://github.com/mahehu/decmeg)
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 69 UPenn and Mayo Clinic's Seizure Detection Challenge
*Detect seizures in intracranial EEG recordings*
  - [Competition overview](https://www.kaggle.com/c/seizure-detection)
  - [Winner (27th) blog/article](https://jonathanstreet.com/blog/seizure-detection-scikit-learn-pipelines/)
  - [Winner (27th) notebook/code/kernel](https://github.com/streety/kaggle-seizure-prediction)
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 70   The Hunt for Prohibited Content
*Predict which ads contain illicit content*
  - [Competition overview](https://www.kaggle.com/c/avito-prohibited-content)
  - [Winner blog/article](https://www.kaggle.com/c/avito-prohibited-content/discussion/10178)
  - [Winner (4th) notebook/code/kernel](https://github.com/ChenglongChen/Kaggle_The_Hunt_for_Prohibited_Content)
  - Other notebook/code/kernel - NA
  - Take home message - NA 
    
## 71 Liberty Mutual Group - Fire Peril Loss Cost
*Predict expected fire losses for insurance policies*
  - [Competition overview](https://www.kaggle.com/c/liberty-mutual-fire-peril)
  - [Winner (3rd) blog/article](https://www.kaggle.com/c/liberty-mutual-fire-peril/discussion/10194#53012)
  - [Winner notebook/code/kernel]()
  - Other notebook/code/kernel - NA
  - Take home message - There's one little trick I used, which I guess others have also done. Instead of predicting the losses directly, I took the **logarithm**, and predicted on that.
 
## 72 Higgs Boson Machine Learning Challenge
*Use the ATLAS experiment to identify the Higgs boson*
  - [Competition overview](https://www.kaggle.com/c/higgs-boson)
  - [Winner blog/article]()
  - [Winner notebook/code/kernel](https://github.com/melisgl/higgsml)
  - [Other (2nd) notebook/code/kernel](https://github.com/TimSalimans/HiggsML)
  - Take home message - NA 
      
## 73 Display Advertising Challenge
*Predict click-through rates on display ads*
  - [Competition overview](https://www.kaggle.com/c/criteo-display-ad-challenge)
  - [Winner blog/article](https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf)
  - [Winner (4th) blog/article](https://github.com/juliandewit/kaggle_criteo/blob/master/ModelDocumentation.pdf)
  - [Winner (2nd) notebook/code/kernel](https://github.com/songgc/display-advertising-challenge)
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 74 CIFAR-10 - Object Recognition in Images
*Identify the subject of 60,000 labeled images*
  - [Competition overview](https://www.kaggle.com/c/cifar-10)
  - [Winner blog/article](https://www.kaggle.com/c/cifar-10/discussion/10493)
  - [Winner notebook/code/kernel](https://github.com/btgraham/SparseConvNet)
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 75 Africa Soil Property Prediction Challenge
*Predict physical and chemical properties of soil using spectral measurements*
  - [Competition overview](https://www.kaggle.com/c/afsis-soil-properties)
  - [Winner (3rd) blog/article](https://www.dropbox.com/sh/hvocwmamojjx20l/AADgzEZ6bp8nlv2jKreK2kUna?dl=0&preview=CodiLime+-+Solution+for+the+AfSIS+competition.pdf)
  - [Winner notebook/code/kernel]()
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 76 Learning Social Circles in Networks
*Model friend memberships to multiple circles*
  - [Competition overview](https://www.kaggle.com/c/learning-social-circles)
  - [Winner blog/article](https://inventingsituations.net/2014/11/09/kaggle-social-networks-competition/)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - It’s well-known that people tend to over-fit the data in the Public leaderboard.  In this case, there were a total of 110 data instances, of which ‘solutions’ were provided for 60.  One third of the remaining 50 instances were used for the Public scoring, and two-thirds were used for the Private scoring.  I got the sense from my work with the Test data that the Public set was a little bit strange, and so I tried to restrain myself from putting too much work into doing well on the Public leaderboard, and instead on understanding and doing well with the Test data.  This seems to have worked well for me in the end.
  
## 77 Tradeshift Text Classification
*Classify text blocks in documents*
  - [Competition overview](https://www.kaggle.com/c/tradeshift-text-classification)
  - Winner blog - NA
  - [Winner notebook/code/kernel](https://github.com/daxiongshu/tradeshift-text-classification)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 78 American Epilepsy Society Seizure Prediction Challenge
*Predict seizures in intracranial EEG recordings*
  - [Competition overview](https://www.kaggle.com/c/seizure-prediction)
  - [Winner blog/article](https://www.kaggle.com/c/seizure-prediction/discussion/11024)
  - [Winner (13th) notebook/code/kernel](https://github.com/udibr/seizure-detection-boost)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 79 Data Science London + Scikit-learn
*Scikit-learn is an open-source machine learning library for Python. Give it a try here!*
  - [Competition overview](https://www.kaggle.com/c/data-science-london-scikit-learn)
  - Winner blog - NA
  - [Winner (7th) notebook/code/kernel](https://github.com/elenacuoco/London-scikit)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 80 Click-Through Rate Prediction
*Predict whether a mobile ad will be clicked*
  - [Competition overview](https://www.kaggle.com/c/avazu-ctr-prediction)
  - [Winner blog/article](https://www.csie.ntu.edu.tw/~r01922136/slides/kaggle-avazu.pdf)
  - [Winner notebook/code/kernel](https://github.com/ycjuan/kaggle-avazu)
  - Other notebook/code/kernel - NA
  - Take home message - **Instead of using the whole dataset**, in this competition we find splitting data into small parts works better than directly using the entire dataset. For example, in one of our models we select instances whose site id is 85f751fd; and in another one we select instances whose app id is ecad2386

## 81 BCI Challenge @ NER 2015
*A spell on you if you cannot detect errors!*
  - [Competition overview](https://www.kaggle.com/c/inria-bci-challenge)
  - Winner blog - NA
  - [Winner notebook/code/kernel](https://github.com/alexandrebarachant/bci-challenge-ner-2015)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 82 Sentiment Analysis on Movie Reviews
*Classify the sentiment of sentences from the Rotten Tomatoes dataset*
  - [Competition overview](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)
  - Winner blog - NA
  - [Winner notebook/code/kernel](https://github.com/rafacarrascosa/samr)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 83 Driver Telematics Analysis
*Use telematic data to identify a driver signature*
  - [Competition overview](https://www.kaggle.com/c/axa-driver-telematics-analysis)
  - [Winner blog/article]()
  - [Winner (2nd) notebook/code/kernel](https://github.com/PrincipalComponent-zz/AXA_Telematics)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 84 National Data Science Bowl
*Predict ocean health, one plankton at a time*
  - [Competition overview](https://www.kaggle.com/c/datasciencebowl)
  - [Winner blog/article](https://benanne.github.io/2015/03/17/plankton.html)
  - [Winner notebook/code/kernel](https://github.com/benanne/kaggle-ndsb)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 85 Finding Elo
*Predict a chess player's FIDE Elo rating from one game*
  - [Competition overview](https://www.kaggle.com/c/finding-elo/overview)
  - [Winner blog/article](https://www.kaggle.com/c/finding-elo/discussion/13008)
  - [Winner notebook/code/kernel](https://github.com/elyase/kaggle-elo)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 86 Microsoft Malware Classification Challenge (BIG 2015)
*Classify malware into families based on file content and characteristics*
  - Type - Classification
  - [Competition overview](https://www.kaggle.com/c/malware-classification)
  - [Winner blog/article](https://github.com/xiaozhouwang/kaggle_Microsoft_Malware/blob/master/Saynotooverfitting.pdf)
  - [Winner notebook/code/kernel](https://github.com/xiaozhouwang/kaggle_Microsoft_Malware/tree/master/kaggle_Microsoft_malware_small)
  - Other notebook/code/kernel - NA
  - Take home message - Cross-validation plays a critical role to overcome overfitting. Out parameter tuning and model selection is based on local cross-validation **rather than the public leaderboard**.
  
## 87 Billion Word Imputation
*Find and impute missing words in the billion word corpus*
  - Type - Missing word (NLP)
  - [Competition overview](https://www.kaggle.com/c/billion-word-imputation)
  - [Winner blog/article](https://www.kaggle.com/c/billion-word-imputation/discussion/14210)
  - [Winner notebook/code/kernel]()
  - Other notebook/code/kernel - NA
  - Take home message - 1) Guessing a word should be done more conservatively if you are not confident you are inserting at the right position. 2) Guessing long words should be done more conservatively than guessing short words.
  
## 88 Restaurant Revenue Prediction
*Predict annual restaurant sales based on objective measurements*
  - Type - Regression
  - [Competition overview]()
  - [Winner blog/article](https://www.kaggle.com/c/restaurant-revenue-prediction/discussion/14066#78118)
  - [Winner (13th) notebook/code/kernel](https://github.com/bensolucky/TFI/)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 89 How Much Did It Rain?
*Predict probabilistic distribution of hourly rain given polarimetric radar measurements*
  - Type - Multclass classification
  - [Competition overview](https://www.kaggle.com/c/how-much-did-it-rain)
  - [Winner blog/article](https://www.kaggle.com/c/how-much-did-it-rain/discussion/14242)
  - [Winner notebook/code/kernel](https://github.com/danzelmo/how-much-did-it-rain)
  - Other notebook/code/kernel - NA
  - Take home message -  I don't have exact CV scores because it **took too much time** to run CV so the following is based on 7% holdout as well as occasional two fold CV.

## 90 Otto Group Product Classification Challenge
*Classify products into the correct category*
  - Type - Multiclass classification
  - [Competition overview](https://www.kaggle.com/c/otto-group-product-classification-challenge)
  - [Winner blog/article](https://www.kaggle.com/c/otto-group-product-classification-challenge/discussion/14335)
  - [Winner notebook/code/kernel]()
  - Other notebook/code/kernel - NA
  - Take home message - Definetely the best algorithms to solve this problem are: Xgboost, NN and KNN. T-sne reduction also helped a lot. Other algorithm have a minor participation on performance. So we **learnt not to discard low performance algorithms*, since they have enough predictive power to improve performance in a 2nd level training.
  
## 91 Walmart Recruiting II: Sales in Stormy Weather
*Predict how sales of weather-sensitive products are affected by snow and rain*
  - Type - Regression
  - [Competition overview](https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather)
  - [Winner blog/article](https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather/discussion/14452)
  - [Winner notebook/code/kernel](https://github.com/threecourse/kaggle-walmart-recruiting-sales-in-stormy-weather)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 92 Facebook Recruiting IV: Human or Robot?
*Predict if an online bid is made by a machine or a human*
  - Type - Classification
  - [Competition overview]()
  - [Winner blog/article](https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot/discussion/14628#81331)
  - [Winner (2nd) notebook/code/kernel](https://github.com/small-yellow-duck/facebook_auction)
  - Other notebook/code/kernel - NA
  - Take home message - Early on I saw that CV was going to be relatively inaccurate so I ended up choosing 500 resamples of different 2/3 – 1/3 splits. This took the standard error on my CV AUC calculation down to about 0.0007 so that I could reasonably have an idea of whether each feature I tested was making a positive or negative difference. The inaccuracy on a single CV fold and the helpful post by T. Scharf, https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot/forums/t/14394/a-basic-look-at-lb-scores suggested that public LB scores were not going to be particularly useful so I only made 3 submissions to avoid any temptation to overfit the public LB - I find this a very difficult thing to do…
  
## 93 West Nile Virus Prediction
*Predict West Nile virus in mosquitos across the city of Chicago*
  - Type - Classification
  - [Competition overview](https://www.kaggle.com/c/predict-west-nile-virus/overview)
  - Winner blog - NA
  - [Winner (2nd) notebook/code/kernel](https://github.com/diefimov/west_nile_virus_2015)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 94 Bag of Words Meets Bags of Popcorn
*Use Google's Word2Vec for movie reviews*
  - Type - Sentiment Analysis
  - [Competition overview](https://www.kaggle.com/c/word2vec-nlp-tutorial)
  - Winner blog - NA
  - [Winner (3rd) notebook/code/kernel](https://github.com/vinhkhuc/kaggle-sentiment-popcorn)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 95 ECML/PKDD 15: Taxi Trip Time Prediction (II)
*Predict the total travel time of taxi trips based on their initial partial trajectories*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/pkdd-15-taxi-trip-time-prediction-ii)
  - [Winner blog/article]()
  - [Winner notebook/code/kernel](https://github.com/hochthom/kaggle-taxi-ii)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 96 ECML/PKDD 15: Taxi Trajectory Prediction (I)
*Predict the destination of taxi trips based on initial partial trajectories*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i)
  - [Winner blog/article]()
  - [Winner notebook/code/kernel](https://github.com/adbrebs/taxi)
  - Other notebook/code/kernel - NA
  - Take home message - NA
 
## 97 Crowdflower Search Results Relevance
*Predict the relevance of search results from eCommerce sites*
  - Type - Ranking
  - [Competition overview](https://www.kaggle.com/c/crowdflower-search-relevance)
  - Winner blog - NA
  - [Winner notebook/code/kernel](https://github.com/ChenglongChen/kaggle-CrowdFlower)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 98 Diabetic Retinopathy Detection
*Identify signs of diabetic retinopathy in eye images*
  - Type - Classification
  - [Competition overview](https://www.kaggle.com/c/diabetic-retinopathy-detection)
  - [Winner (2nd) blog/article](https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15807)
  - [Winner notebook/code/kernel](https://github.com/sveitser/kaggle_diabetic)
  - Other notebook/code/kernel - NA
  - Take home message - NA  

##  99 Avito Context Ad Clicks
*Predict if context ads will earn a user's click*
  - Type - NA
  - [Competition overview](https://www.kaggle.com/c/avito-context-ad-clicks)
  - Winner blog - NA
  - [Winner (2nd) notebook/code/kernel](https://github.com/Gzsiceberg/kaggle-avito)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 100 ICDM 2015: Drawbridge Cross-Device Connections
*Identify individual users across their digital devices*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/icdm-2015-drawbridge-cross-device-connections)
  - [Winner blog/article](https://www.kaggle.com/c/icdm-2015-drawbridge-cross-device-connections/discussion/16122)
  - [Winner notebook/code/kernel]()
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 101 Liberty Mutual Group: Property Inspection Prediction
*Quantify property hazards before time of inspection*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction)
  - [Winner blog/article]()
  - [Winner (16th) notebook/code/kernel](https://github.com/far0n/kaggle-lmgpip)
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 102 Grasp-and-Lift EEG Detection
*Identify hand motions from EEG recordings*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/grasp-and-lift-eeg-detection)
  - [Winner blog/article](https://github.com/alexandrebarachant/Grasp-and-lift-EEG-challenge)
  - [Winner notebook/code/kernel](https://github.com/alexandrebarachant/Grasp-and-lift-EEG-challenge)
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 103 Coupon Purchase Prediction
*Predict which coupons a customer will buy*
  - Type - Classification
  - [Competition overview](https://www.kaggle.com/c/coupon-purchase-prediction)
  - [Winner blog/article](https://www.kaggle.com/c/coupon-purchase-prediction/discussion/16736#93760)
  - [Winner (3rd) notebook/code/kernel](https://github.com/threecourse/kaggle-coupon-purchase-prediction)
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 104 Flavours of Physics: Finding τ → μμμ
*Identify a rare decay phenomenon*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/flavours-of-physics)
  - [Winner blog/article]()
  - [Winner notebook/code/kernel](https://github.com/yandexdataschool/flavours-of-physics-start)
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 105 Truly Native?
*Predict which web pages served by StumbleUpon are sponsored*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/dato-native)
  - [Winner blog/article](https://www.kaggle.com/c/dato-native/discussion/17009#96206)
  - [Winner notebook/code/kernel]()
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 106 Springleaf Marketing Response
*Determine whether to send a direct mail piece to a customer *
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/springleaf-marketing-response)
  - [Winner (7th) blog/article](https://www.kaggle.com/c/springleaf-marketing-response/discussion/17081#96804)
  - [Winner notebook/code/kernel]()
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 107 How Much Did It Rain? II
*Predict hourly rainfall using data from polarimetric radars*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/how-much-did-it-rain-ii)
  - [Winner blog/article]()
  - [Winner notebook/code/kernel](https://github.com/simaaron/kaggle-Rain)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 108 Rossmann Store Sales
*Forecast sales using store, promotion, and competitor data*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/rossmann-store-sales)
  - [Winner blog/article](https://www.kaggle.com/c/rossmann-store-sales/discussion/18024)
  - [Winner (3rd) notebook/code/kernel](https://github.com/entron/entity-embedding-rossmann)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 109 What's Cooking?
*Use recipe ingredients to categorize the cuisine*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/whats-cooking)
  - Winner blog - NA
  - [Winner (4th) notebook/code/kernel](https://github.com/dmcgarry/kaggle_cooking)
  - Other notebook/code/kernel - NA
  - Take home message - NA  
  
## 110 Walmart Recruiting: Trip Type Classification
*Use market basket analysis to classify shopping trips*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/walmart-recruiting-trip-type-classification)
  - [Winner blog/article]()
  - [Winner (11th) notebook/code/kernel](https://github.com/abhishekkrthakur/walmart2015)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 111 Right Whale Recognition
*Identify endangered right whales in aerial photographs *
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/noaa-right-whale-recognition)
  - [Winner blog/article](https://www.kaggle.com/c/noaa-right-whale-recognition/discussion/18409)
  - [Winner (2nd) notebook/code/kernel](https://github.com/felixlaumon/kaggle-right-whale)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 112 The Winton Stock Market Challenge
*Join a multi-disciplinary team of research scientists*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/the-winton-stock-market-challenge)
  - [Winner (2nd) blog/article](https://www.kaggle.com/c/the-winton-stock-market-challenge/discussion/18584#116236)
  - Winner notebook/code/kernel -NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 113 Cervical Cancer Screening
*Help prevent cervical cancer by identifying at-risk populations*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/cervical-cancer-screening)
  - [Winner blog/article](https://www.kaggle.com/c/cervical-cancer-screening/discussion/18691)
  - [Winner notebook/code/kernel]()
  - Other notebook/code/kernel - NA
  - Take home message - The single most important piece of data was actually not in the dataset. Was what Wendy said very early in the competition: "We filtered out some records that are too relevant, records that is direct evidence that the person is a cervical cancer screener or not." Now, I do not argue the skills of those preparing the dataset, but it's really hard to remove records from a relational database without leaving bread crumbs behind you. So, I set myself to look for those crumbs while I was modelling other things I knew had predictive power.
  
## 114 Homesite Quote Conversion
*Which customers will purchase a quoted insurance plan?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/homesite-quote-conversion)
  - Winner blog - NA
  - [Winner notebook/code/kernel](https://github.com/Far0n/kaggle-homesite)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 115 Airbnb New User Bookings
*Where will a new guest book their first travel experience?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings)
  - [Winner blog/article]()
  - [Winner (3rd) notebook/code/kernel](https://github.com/svegapons/kaggle_airbnb)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 116 The Allen AI Science Challenge
*Is your model smarter than an 8th grader?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/the-allen-ai-science-challenge)
  - [Winner blog/article]()
  - [Winner notebook/code/kernel](https://github.com/Cardal/Kaggle_AllenAIscience)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 117 Prudential Life Insurance Assessment
*Can you make buying life insurance easier?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/prudential-life-insurance-assessment)
  - [Winner blog/article](https://www.kaggle.com/c/prudential-life-insurance-assessment/discussion/19010)
  - [Winner (2nd) notebook/code/kernel](https://github.com/zhurak/kaggle-prudential)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 118 Telstra Network Disruptions
*Predict service faults on Australia's largest telecommunications network*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/telstra-recruiting-network)
  - [Winner blog/article]()
  - [Winner (7th) notebook/code/kernel](https://github.com/gaborfodor/TNP)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 119 Second Annual Data Science Bowl
*Transforming How We Diagnose Heart Disease*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/second-annual-data-science-bowl)
  - [Winner blog/article]()
  - [Winner notebook/code/kernel](https://github.com/woshialex/diagnose-heart)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 120 Yelp Restaurant Photo Classification
*Predict attribute labels for restaurants using user-submitted photos*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/yelp-restaurant-photo-classification)
  - [Winner blog/article]()
  - [Winner notebook/code/kernel](https://github.com/u1234x1234/kaggle-yelp-restaurant-photo-classification)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 121 BNP Paribas Cardif Claims Management
*Can you accelerate BNP Paribas Cardif's claims management process?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/overview)
  - [Winner blog/article](https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/discussion/20247)
  - [Winner (2nd) notebook/code/kernel](https://github.com/bishwarup307/BNP_Paribas_Cardiff_Claim_Management)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 122 Home Depot Product Search Relevance
*Predict the relevance of search results on homedepot.com*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/home-depot-product-search-relevance)
  - [Winner blog/article](https://www.kaggle.com/c/home-depot-product-search-relevance/discussion/20427#116960)
  - [Winner (3rd) notebook/code/kernel](https://github.com/ChenglongChen/kaggle-HomeDepot)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 123 Santander Customer Satisfaction
*Which customers are happy customers?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/santander-customer-satisfaction)
  - [Winner blog/article](https://www.kaggle.com/c/santander-customer-satisfaction/discussion/20647#118259)
  - [Winner (3rd) notebook/code/kernel](https://github.com/diefimov/santander_2016)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 124 San Francisco Crime Classification
*Predict the category of crimes that occurred in the city by the bay *
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/sf-crime)
  - Winner blog - NA
  - [Winner notebook/code/kernel](https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 125 Expedia Hotel Recommendations
*Which hotel type will an Expedia customer book?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/expedia-hotel-recommendations)
  - [Winner blog/article](https://www.kaggle.com/c/expedia-hotel-recommendations/discussion/21607)
  - Winner notebook/code/kernel - NA
  - [Other notebook/code/kernel](https://github.com/mandalsubhajit/Kaggle--Expedia-Hotel-Recommendations)
  - Take home message - NA

## 126 Kobe Bryant Shot Selection
*Which shots did Kobe sink?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/kobe-bryant-shot-selection)
  - Winner blog -NA
  - [Winner (12th) notebook/code/kernel](https://github.com/shiba24/kaggle-Kobe-Bryant-Shot-Selection)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 127 Draper Satellite Image Chronology
*Can you put order to space and time?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/draper-satellite-image-chronology)
  - [Winner blog/article](https://www.kaggle.com/c/draper-satellite-image-chronology/discussion/21936)
  - [Winner (12th) notebook/code/kernel](https://github.com/mingtotti/kaggle/tree/master/draper)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 128 Facebook V: Predicting Check Ins
*Identify the correct place for check ins*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/facebook-v-predicting-check-ins)
  - [Winner (7th) blog/article](https://www.kaggle.com/c/facebook-v-predicting-check-ins/discussion/22078#126195)
  - [Winner (2nd) notebook/code/kernel](https://github.com/mkliegl/kaggle-Facebook-V)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 129 Avito Duplicate Ads Detection
*Can you detect duplicitous duplicate ads?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/avito-duplicate-ads-detection)
  - [Winner (2nd) blog/article](https://www.kaggle.com/c/avito-duplicate-ads-detection/discussion/22205)
  - [Winner (2nd) notebook/code/kernel](https://github.com/sonnylaskar/Competitions/tree/master/Kaggle/Avito%20Duplicate%20Ad%20Detection)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 130 State Farm Distracted Driver Detection
*Can computer vision spot distracted drivers?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/state-farm-distracted-driver-detection)
  - [Winner blog/article](https://www.kaggle.com/c/state-farm-distracted-driver-detection/discussion/22906)
  - [Winner (29th) notebook/code/kernel](https://github.com/oswaldoludwig/Human-Action-Recognition-with-Keras)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 131 Ultrasound Nerve Segmentation
*Identify nerve structures in ultrasound images of the neck*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/ultrasound-nerve-segmentation)
  - [Winner blog/article]()
  - [Winner (25th) notebook/code/kernel](https://github.com/julienr/kaggle_uns)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 132 Grupo Bimbo Inventory Demand
*Maximize sales and minimize returns of bakery goods*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/grupo-bimbo-inventory-demand)
  - [Winner blog/article](https://www.kaggle.com/c/grupo-bimbo-inventory-demand/discussion/23863)
  - [Winner notebook/code/kernel]()
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 133 TalkingData Mobile User Demographics
*Get to know millions of mobile device users*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/talkingdata-mobile-user-demographics)
  - [Winner blog/article](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56475)
  - [Winner (3rd) notebook/code/kernel](https://github.com/chechir/talking_data)
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 134 Predicting Red Hat Business Value
*Classify customer potential*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/predicting-red-hat-business-value)
  - [Winner blog/article](https://www.kaggle.com/c/predicting-red-hat-business-value/discussion/23786)
  - [Winner notebook/code/kernel]()
  - Other notebook/code/kernel - NA
  - Take home message - First, proper cross-validation set was very important. Tricks I did to have a representative CV set.
  
## 135 Integer Sequence Learning
*1, 2, 3, 4, 5, 7?!*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/integer-sequence-learning)
  - [Winner (17th) blog/article](https://www.kaggle.com/c/integer-sequence-learning/discussion/24971)
  - [Winner (?) notebook/code/kernel](https://github.com/Kyubyong/integer_sequence_learning)
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 136 Painter by Numbers
*Does every painter leave a fingerprint?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/painter-by-numbers)
  - Winner blog - NA
  - [Winner notebook/code/kernel](https://github.com/inejc/painters)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 137 Bosch Production Line Performance
*Reduce manufacturing failures*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/bosch-production-line-performance)
  - [Winner blog/article](https://www.kaggle.com/c/bosch-production-line-performance/discussion/25434)
  - [Winner (57) notebook/code/kernel](https://github.com/toshi-k/kaggle-bosch-production-line-performance)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 138 Melbourne University AES/MathWorks/NIH Seizure Prediction
*Predict seizures in long-term human intracranial EEG recordings*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/melbourne-university-seizure-prediction)
  - [Winner blog/article](https://www.kaggle.com/c/melbourne-university-seizure-prediction/discussion/26310)
  - [Winner notebook/code/kernel](https://github.com/alexandrebarachant/kaggle-seizure-prediction-challenge-2016)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 139 Allstate Claims Severity
*How severe is an insurance claim?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/allstate-claims-severity)
  - [Winner blog/article](https://www.kaggle.com/c/allstate-claims-severity/discussion/26416)
  - [Winner (2nd) notebook/code/kernel](https://github.com/alno/kaggle-allstate-claims-severity)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 140 Santander Product Recommendation
*Can you pair products with people?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/santander-product-recommendation)
  - [Winner (2nd) blog/article](https://ttvand.github.io/Second-place-in-the-Santander-product-Recommendation-Kaggle-competition/)
  - [Winner (2nd) notebook/code/kernel](https://github.com/ttvand/Santander-Product-Recommendation)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 141 Facial Keypoints Detection
*Detect the location of keypoints on face images*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/facial-keypoints-detection)
  - [Winner blog/article](https://www.kaggle.com/c/facial-keypoints-detection/discussion/21766)
  - Winner notebook/code/kernel - NA
  - [Other notebook/code/kernel](https://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/)
  - Take home message - NA

## 142 Outbrain Click Prediction
*Can you predict which recommended content each user will click?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/outbrain-click-prediction)
  - Winner blog - NA
  - [Winner (13th) notebook/code/kernel](https://github.com/alexeygrigorev/outbrain-click-prediction-kaggle)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 143 Two Sigma Financial Modeling Challenge
*Can you uncover predictive value in an uncertain world?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/two-sigma-financial-modeling)
  - [Winner (7th) blog/article](https://www.kaggle.com/c/two-sigma-financial-modeling/discussion/29793)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 144 Dstl Satellite Imagery Feature Detection
*Can you train an eye in the sky?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection)
  - [Winner (2nd) blog/article](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/discussion/29747)
  - [Winner (3rd) notebook/code/kernel](https://github.com/ternaus/kaggle_dstl_submission)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 145 Transfer Learning on Stack Exchange Tags
*Predict tags from models trained on unrelated topics*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags)
  - Winner blog - NA
  - [Winner (4th) notebook/code/kernel](https://github.com/viig99/stackexchange-transfer-learning)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 146 Data Science Bowl 2017
*Can you improve lung cancer detection?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/data-science-bowl-2017)
  - Winner blog - NA
  - [Winner notebook/code/kernel](https://github.com/lfz/DSB2017)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 147 The Nature Conservancy Fisheries Monitoring
*Can you detect and classify species of fish?*
  - Type - Classification
  - [Competition overview](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring)
  - [Winner blog/article](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/discussion/31538#175210)
  - [Winner (?) notebook/code/kernel](https://github.com/tchaton/The-Nature-Conservancy-Fisheries-Monitoring-Challenge)
  - Other notebook/code/kernel - NA
  - Take home message - NA
 
## 148 Two Sigma Connect: Rental Listing Inquiries
*How much interest will a new rental listing on RentHop receive?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries)
  - [Winner (2nd) blog/article](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32148)
  - [Winner notebook/code/kernel](https://github.com/plantsgo/Rental-Listing-Inquiries)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 149 Google Cloud & YouTube-8M Video Understanding Challenge
*Can you produce the best video tag predictions?*
  - Type - Video Tagging
  - [Competition overview](https://www.kaggle.com/c/youtube8m)
  - [Winner blog/article](https://www.kaggle.com/c/youtube8m/discussion/35063)
  - [Winner notebook/code/kernel](https://github.com/antoine77340/Youtube-8M-WILLOW)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 150 Quora Question Pairs
*Can you identify question pairs that have the same intent?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/quora-question-pairs)
  - [Winner blog/article](https://www.kaggle.com/c/quora-question-pairs/discussion/34355)
  - [Winner (3rd) notebook/code/kernel](https://github.com/sjvasquez/quora-duplicate-questions)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 151 Intel & MobileODT Cervical Cancer Screening
*Which cancer treatment will be most effective?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening)
  - Winner blog - NA
  - [Winner (4th) notebook/code/kernel](https://github.com/lRomul/intel-cancer)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 152 NOAA Fisheries Steller Sea Lion Population Count
*How many sea lions do you see?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count)
  - Winner blog - NA
  - [Winner notebook/code/kernel](https://www.kaggle.com/outrunner/use-keras-to-count-sea-lions/notebook)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 153 Sberbank Russian Housing Market
*Can you predict realty price fluctuations in Russia’s volatile economy?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/sberbank-russian-housing-market)
  - [Winner blog/article](https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/35684)
  - [Winner (15th) notebook/code/kernel](https://github.com/Danila89/sberbank_kaggle)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 154 iNaturalist Challenge at FGVC 2017
*Fine-grained classification challenge spanning 5,000 species.*
  - Type - Classification
  - [Competition overview](https://www.kaggle.com/c/inaturalist-challenge-at-fgvc-2017)
  - Winner blog - NA
  - [Winner (13th) notebook/code/kernel](https://github.com/phunterlau/iNaturalist)
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 155 Mercedes-Benz Greener Manufacturing
*Can you cut the time a Mercedes-Benz spends on the test bench?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing)
  - [Winner blog/article](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/37700)
  - [Winner (11th) notebook/code/kernel](https://github.com/Danila89/kaggle_mercedes)
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 156 Planet: Understanding the Amazon from Space
*Use satellite data to track the human footprint in the Amazon rainforest*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)
  - [Winner blog/article](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36809)
  - [Winner (3rd) notebook/code/kernel](https://github.com/ZFTurbo/Kaggle-Planet-Understanding-the-Amazon-from-Space)
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 157 Instacart Market Basket Analysis
*Which products will an Instacart consumer purchase again?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/instacart-market-basket-analysis)
  - Winner blog - NA
  - [Winner (2nd) notebook/code/kernel](https://github.com/KazukiOnodera/Instacart)
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 158 Invasive Species Monitoring
*Identify images of invasive hydrangea*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/invasive-species-monitoring)
  - Winner blog - NA
  - [Winner notebook/code/kernel](https://github.com/jamesrequa/Kaggle-Invasive-Species-Monitoring-Competition)
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 159 New York City Taxi Trip Duration
*Share code and data to improve ride time predictions*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/nyc-taxi-trip-duration)
  - [Winner (4th) blog/article](https://www.kaggle.com/c/nyc-taxi-trip-duration/discussion/39553)
  - Winner notebook/code/kernel
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 160 Carvana Image Masking Challenge
*Automatically identify the boundaries of the car in an image*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/carvana-image-masking-challenge)
  - [Winner blog/article](https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/46176)
  - [Winner notebook/code/kernel](https://github.com/asanakoy/kaggle_carvana_segmentation)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 161 NIPS 2017: Defense Against Adversarial Attack
*Create an image classifier that is robust to adversarial attacks*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/nips-2017-defense-against-adversarial-attack)
  - Winner blog - NA
  - [Winner (3rd) notebook/code/kernel](https://github.com/anlthms/nips-2017)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 162 NIPS 2017: Targeted Adversarial Attack
*Develop an adversarial attack that causes image classifiers to predict a specific target class*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/nips-2017-targeted-adversarial-attack)
  - Winner blog - NA
  - [Winner notebook/code/kernel](https://github.com/dongyp13/Targeted-Adversarial-Attack)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 163 NIPS 2017: Non-targeted Adversarial Attack
*Imperceptibly transform images in ways that fool classification models*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack)
  - Winner blog - NA
  - [Winner notebook/code/kernel](https://github.com/dongyp13/Non-Targeted-Adversarial-Attacks)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 164 Web Traffic Time Series Forecasting
*Forecast future traffic to Wikipedia pages*
  - Type - Forecasting
  - [Competition overview](https://www.kaggle.com/c/web-traffic-time-series-forecasting)
  - [Winner blog/article](https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/43795)
  - [Winner (2nd) notebook/code/kernel](https://github.com/jfpuget/Kaggle/tree/master/WebTrafficPrediction)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 165 Text Normalization Challenge - Russian Language
*Convert Russian text from written expressions into spoken forms*
  - Type - NLP
  - [Competition overview](https://www.kaggle.com/c/text-normalization-challenge-russian-language)
  - Winner blog - NA
  - [Winner (3rd) notebook/code/kernel](https://github.com/ppleskov/Text-Normalization-Challenge-Russian-Language)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 166 Text Normalization Challenge - English Language
*Convert English text from written expressions into spoken forms*
  - Type - NLP
  - [Competition overview](https://www.kaggle.com/c/text-normalization-challenge-english-language)
  - [Winner (4th) blog/article](https://www.kaggle.com/c/text-normalization-challenge-english-language/discussion/43963)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 167 Porto Seguro’s Safe Driver Prediction
*Predict if a driver will file an insurance claim next year.*
  - Type - Classification
  - [Competition overview](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)
  - [Winner blog/article](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629)
  - [Winner (2nd) notebook/code/kernel](https://github.com/xiaozhouwang/kaggle-porto-seguro)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 168 Cdiscount’s Image Classification Challenge
*Categorize e-commerce photos*
  - Type - Classification
  - [Competition overview](https://www.kaggle.com/c/cdiscount-image-classification-challenge)
  - [Winner blog/article](https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/45863)
  - [Winner (2nd) notebook/code/kernel](https://github.com/miha-skalic/convolutedPredictions_Cdiscount)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 169 Spooky Author Identification
*Share code and discuss insights to identify horror authors from their writings*
  - Type - NLP
  - [Competition overview](https://www.kaggle.com/c/spooky-author-identification)
  - Winner blog - NA
  - Winner notebook/code/kernel - NA
  - [Other notebook/code/kernel](https://github.com/pkubik/horror)
  - Take home message - NA

## 170 Passenger Screening Algorithm Challenge
*Improve the accuracy of the Department of Homeland Security's threat recognition algorithms*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/passenger-screening-algorithm-challenge)
  - [Winner blog/article](https://www.kaggle.com/c/passenger-screening-algorithm-challenge/discussion/45805#261052)
  - [Winner (7th) notebook/code/kernel](https://github.com/suchir/passenger_screening_algorithm_challenge)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 171 WSDM - KKBox's Music Recommendation Challenge
*Can you build the best music recommendation system?*
  - Type - Recommender System
  - [Competition overview](https://www.kaggle.com/c/kkbox-music-recommendation-challenge)
  - [Winner blog/article](https://www.kaggle.com/c/kkbox-music-recommendation-challenge/discussion/45942)
  - [Winner (3rd) notebook/code/kernel](https://github.com/VasiliyRubtsov/wsdm_music_recommendations)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 172 Zillow Prize: Zillow’s Home Value Prediction (Zestimate)
*Can you improve the algorithm that changed the world of real estate?*
  - Type - Regression
  - [Competition overview](https://www.kaggle.com/c/zillow-prize-1)
  - [Winner (17th) blog/article](https://www.kaggle.com/c/zillow-prize-1/discussion/47434)
  - [Winner notebook/code/kernel]()
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 173 Santa Gift Matching Challenge
*Down through the chimney with lots of toys...*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/santa-gift-matching)
  - [Winner (2nd) blog/article](https://www.kaggle.com/c/santa-gift-matching/discussion/47386)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 174 Corporación Favorita Grocery Sales Forecasting
*Can you accurately predict sales for a large grocery chain?*
  - Type - Regression
  - [Competition overview](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)
  - [Winner blog/article](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47582)
  - Winner notebook/code/kernel - see winner blog
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 175 TensorFlow Speech Recognition Challenge
*Can you build an algorithm that understands simple speech commands?*
  - Type - Speech Recognition
  - [Competition overview](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge)
  - [Winner blog/article](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/46988)
  - [Winner (3rd) notebook/code/kernel](https://github.com/xiaozhouwang/tensorflow_speech_recognition_solution)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 176 Statoil/C-CORE Iceberg Classifier Challenge
*Ship or iceberg, can you decide from space?*
  - Type - Classification
  - [Competition overview](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge)
  - [Winner blog/article](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/discussion/48241)
  - Winner notebook/code/kernel - NA
  - [Other notebook/code/kernel](https://github.com/cttsai1985/Kaggle-Statoil-Iceberg-Classifier-ConvNets)
  - Take home message - NA
  
## 177 Recruit Restaurant Visitor Forecasting
*Predict how many future visitors a restaurant will receive*
  - Type - Regression
  - [Competition overview](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting)
  - [Winner blog/article](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/discussion/49129)
  - [Winner (3rd) notebook/code/kernel](https://github.com/dkivaranovic/kaggledays-recruit)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 178 IEEE's Signal Processing Society - Camera Model Identification
*Identify from which camera an image was taken*
  - Type - Classification
  - [Competition overview](https://www.kaggle.com/c/sp-society-camera-model-identification)
  - [Winner blog/article](https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/49367)
  - [Winner (2nd) notebook/code/kernel](https://github.com/ikibardin/kaggle-camera-model-identification)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 179 Nomad2018 Predicting Transparent Conductors
*Predict the key properties of novel transparent semiconductors*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/nomad2018-predict-transparent-conductors)
  - [Winner (5th) blog/article](https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/discussion/49903)
  - [Winner notebook/code/kernel]()
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 180 Mercari Price Suggestion Challenge
*Can you automatically suggest product prices to online sellers?*
  - Type - Recommender System
  - [Competition overview](https://www.kaggle.com/c/mercari-price-suggestion-challenge)
  - [Winner blog/article](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/50256)
  - [Winner notebook/code/kernel](https://github.com/pjankiewicz/mercari-solution)
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 181 Toxic Comment Classification Challenge
*Identify and classify toxic online comments*
  - Type - Classification
  - [Competition overview](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
  - [Winner blog/article](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 182 Google Cloud & NCAA® ML Competition 2018-Women's
*Apply machine learning to NCAA® March Madness®*
  - Type - Forecasting
  - [Competition overview](https://www.kaggle.com/c/womens-machine-learning-competition-2018)
  - [Winner blog/article](https://www.kaggle.com/c/womens-machine-learning-competition-2018/discussion/53597)
  - Winner notebook/code/kernel -NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 183 Google Cloud & NCAA® ML Competition 2018-Men's
*Apply Machine Learning to NCAA® March Madness®*
  - Type - Forecasting
  - [Competition overview](https://www.kaggle.com/c/mens-machine-learning-competition-2018)
  - [Winner blog/article](https://www.kaggle.com/c/womens-machine-learning-competition-2019/discussion/80689#latest-509892)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 184 2018 Data Science Bowl
*Find the nuclei in divergent images to advance medical discovery*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/data-science-bowl-2018)
  - Winner blog - NA
  - [Winner notebook/code/kernel](https://github.com/selimsef/dsb2018_topcoders)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 185 DonorsChoose.org Application Screening
*Predict whether teachers' project proposals are accepted*
  - Type - NLP
  - [Competition overview](https://www.kaggle.com/c/donorschoose-application-screening)
  - [Winner blog/article](https://www.kaggle.com/shadowwarrior/1st-place-solution/notebook)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 186 TalkingData AdTracking Fraud Detection Challenge
*Can you detect fraudulent click traffic for mobile app ads?*
  - Type - Classification
  - [Competition overview](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection)
  - [Winner blog/article](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56475)
  - [Winner notebook/code/kernel](https://github.com/flowlight0/talkingdata-adtracking-fraud-detection)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 187 Google Landmark Retrieval Challenge
*Given an image, can you find all of the same landmarks in a dataset?*
  - Type - Information Retrieval
  - [Competition overview](https://www.kaggle.com/c/landmark-retrieval-challenge)
  - [Winner blog/article](https://www.kaggle.com/c/landmark-retrieval-challenge/discussion/57855)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 188 Google Landmark Recognition Challenge
*Label famous (and not-so-famous) landmarks in images*
  - Type - Labelling
  - [Competition overview](https://www.kaggle.com/c/landmark-recognition-challenge)
  - [Winner blog/article](https://www.kaggle.com/c/landmark-recognition-challenge/discussion/57896)
  - [Winner (19th) notebook/code/kernel](https://github.com/jandaldrop/landmark-recognition-challenge)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 189 iMaterialist Challenge (Furniture) at FGVC5
*Image Classification of Furniture & Home Goods.*
  - Type - Classification
  - [Competition overview](https://www.kaggle.com/c/imaterialist-challenge-furniture-2018)
  - [Winner blog/article](https://www.kaggle.com/c/imaterialist-challenge-furniture-2018/discussion/57951)
  - Winner notebook/code/kernel
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 190 iMaterialist Challenge (Fashion) at FGVC5
*Image classification of fashion products.*
  - Type - Image Classification
  - [Competition overview](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018)
  - [Winner blog/article](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/discussion/57944)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 191 CVPR 2018 WAD Video Segmentation Challenge
*Can you segment each objects within image frames captured by vehicles?*
  - Type - Video Segmentation
  - [Competition overview](https://www.kaggle.com/c/cvpr-2018-autonomous-driving)
  - Winner blog - NA
  - [Winner (2nd) notebook/code/kernel](https://github.com/Computational-Camera/Kaggle-CVPR-2018-WAD-Video-Segmentation-Challenge-Solution)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 192 Avito Demand Prediction Challenge
*Predict demand for an online classified ad*
  - Type - Forecasting
  - [Competition overview](https://www.kaggle.com/c/avito-demand-prediction)
  - [Winner blog/article](https://www.kaggle.com/c/avito-demand-prediction/discussion/59880)
  - [Winner (5th) notebook/code/kernel](https://github.com/darraghdog/avito-demand)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 193 Freesound General-Purpose Audio Tagging Challenge
*Can you automatically recognize sounds from a wide range of real-world environments?*
  - Type - Audio Tagging
  - [Competition overview](https://www.kaggle.com/c/freesound-audio-tagging)
  - [Winner (4th) blog/article](https://www.kaggle.com/c/freesound-audio-tagging/discussion/62634)
  - [Winner (4th) notebook/code/kernel](https://github.com/Cocoxili/DCASE2018Task2)
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 194 The 2nd YouTube-8M Video Understanding Challenge
*Can you create a constrained-size model to predict video labels?*
  - Type - Video Labelling
  - [Competition overview](https://www.kaggle.com/c/youtube8m-2018)
  - [Winner blog/article](https://www.kaggle.com/c/youtube8m-2018/discussion/62781)
  - [Winner notebook/code/kernel](https://github.com/miha-skalic/youtube8mchallenge)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 195 TrackML Particle Tracking Challenge
*High Energy Physics particle tracking in CERN detectors*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/trackml-particle-identification)
  - [Winner blog/article](https://www.kaggle.com/c/trackml-particle-identification/discussion/63249)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 196 Santander Value Prediction Challenge
*Predict the value of transactions for potential customers.*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/santander-value-prediction-challenge)
  - [Winner blog/article](https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/63907)
  - [Winner (21st) notebook/code/kernel](https://www.kaggle.com/rsakata/21st-place-solution-bug-fixed-private-0-52785/code)
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 197 Home Credit Default Risk
*Can you predict how capable each applicant is of repaying a loan?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/home-credit-default-risk)
  - [Winner blog/article](https://www.kaggle.com/c/home-credit-default-risk/discussion/64821)
  - [Winner (2nd) notebook/code/kernel](https://github.com/KazukiOnodera/Home-Credit-Default-Risk)
  - Other notebook/code/kernel - NA
  - Take home message - NA
 
## 198 Google AI Open Images - Object Detection Track
*Detect objects in varied and complex images.*
  - Type - Object Detection
  - [Competition overview](https://www.kaggle.com/c/google-ai-open-images-object-detection-track)
  - [Winner (2nd) blog/article](https://arxiv.org/pdf/1809.00778.pdf)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 199 Google AI Open Images - Visual Relationship Track
*Detect pairs of objects in particular relationships.*
  - Type - Object Detection
  - [Competition overview](https://www.kaggle.com/c/google-ai-open-images-visual-relationship-track)
  - [Winner blog/article](https://www.kaggle.com/c/google-ai-open-images-visual-relationship-track/discussion/64651)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 200 TGS Salt Identification Challenge
*Segment salt deposits beneath the Earth's surface*
  - Type - Image Segmentation
  - [Competition overview]()
  - [Winner blog/article](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69291)
  - [Winner (4th) notebook/code/kernel](https://github.com/ybabakhin/kaggle_salt_bes_phalanx)
  - [Other notebook/code/kernel](https://github.com/SeuTao/TGS-Salt-Identification-Challenge-2018-_4th_place_solution))
  - Take home message - NA  

## 201 RSNA Pneumonia Detection Challenge
*Can you build an algorithm that automatically detects potential pneumonia cases?*
  - Type - Image Classification
  - [Competition overview](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
  - Winner blog - NA
  - [Winner notebook/code/kernel](https://github.com/i-pan/kaggle-rsna18)
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 202 Inclusive Images Challenge
*Stress test image classifiers across new geographic distributions*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/inclusive-images-challenge)
  - [Winner (2nd) blog/article](https://www.kaggle.com/c/inclusive-images-challenge/discussion/72450)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 203 Airbus Ship Detection Challenge
*Find ships on satellite images as quickly as possible*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/airbus-ship-detection)
  - [Winner blog/article](https://www.kaggle.com/c/airbus-ship-detection/discussion/74443)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
 ## 204 Don't call me turkey!
*Thanksgiving Edition: Find the turkey in the sound bite*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/dont-call-me-turkey)
  - Winner blog - NA
  - [Winner (2nd) notebook/code/kernel](https://github.com/yfe404/kaggle-1st-place-dont-call-me-turkey)
  - Other notebook/code/kernel - NA
  - Take home message - NA  
     
## 205 Quick, Draw! Doodle Recognition Challenge
*How accurately can you identify a doodle?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/quickdraw-doodle-recognition)
  - [Winner blog/article](https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/73738)
  - [Winner (8th) notebook/code/kernel](https://github.com/alekseynp/kaggle-quickdraw)
  - Other notebook/code/kernel - NA
  - Take home message - NA  
  
## 206 PLAsTiCC Astronomical Classification
*Can you help make sense of the Universe?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/PLAsTiCC-2018)
  - [Winner blog/article](https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75033)
  - [Winner notebook/code/kernel](https://github.com/kboone/avocado)
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 207 Traveling Santa 2018 - Prime Paths
*But does your code recall, the most efficient route of all?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/traveling-santa-2018-prime-paths)
  - [Winner (2nd) blog/article](https://www.kaggle.com/c/traveling-santa-2018-prime-paths/discussion/77250)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  
  
## 208 Human Protein Atlas Image Classification
*Classify subcellular protein patterns in human cells*
  - Type - Classification
  - [Competition overview](https://www.kaggle.com/c/human-protein-atlas-image-classification)
  - [Winner blog/article](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109)
  - [Winner (3rd) notebook/code/kernel](https://github.com/pudae/kaggle-hpa)
  - Other notebook/code/kernel - NA
  - Take home message - NA  
  
## 209 20 Newsgroups Ciphertext Challenge
*V8g{9827$A${?^*?}$$v7*.yig$w9.8}*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/20-newsgroups-ciphertext-challenge)
  - [Winner blog/article](https://www.kaggle.com/c/20-newsgroups-ciphertext-challenge/discussion/77894)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  
  
## 210 PUBG Finish Placement Prediction (Kernels Only)
*Can you predict the battle royale finish of PUBG Players?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/pubg-finish-placement-prediction)
  - [Winner blog/article](https://www.kaggle.com/c/pubg-finish-placement-prediction/discussion/79161)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 211  Reducing Commercial Aviation Fatalities
*Can you tell when a pilot is heading for trouble?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/reducing-commercial-aviation-fatalities)
  - [Winner (8th) blog/article](https://www.kaggle.com/c/reducing-commercial-aviation-fatalities/discussion/84527)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  
  
## 212 Quora Insincere Questions Classification
*Detect toxic content to improve online conversations*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/quora-insincere-questions-classification)
  - [Winner blog/article](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80568)
  - [Winner notebook/code/kernel](https://www.kaggle.com/wowfattie/3rd-place)
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 213 Google Analytics Customer Revenue Prediction
*Predict how much GStore customers will spend*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/ga-customer-revenue-prediction)
  - [Winner blog/article](https://www.kaggle.com/c/ga-customer-revenue-prediction/discussion/82614)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  
  
## 214 Elo Merchant Category Recommendation
*Help understand customer loyalty*
  - Type - Category Recommendation
  - [Competition overview](https://www.kaggle.com/c/elo-merchant-category-recommendation)
  - [Winner blog/article](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82036)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 215 Humpback Whale Identification
*Can you identify a whale by its tail?*
  - Type - Images Classification
  - [Competition overview](https://www.kaggle.com/c/humpback-whale-identification)
  - [Winner blog/article](https://www.kaggle.com/c/humpback-whale-identification/discussion/82366#latest-523382)
  - [Winner notebook/code/kernel](https://github.com/earhian/Humpback-Whale-Identification-1st-)
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 216 Microsoft Malware Prediction
*Can you predict if a machine will soon be hit with malware?*
  - Type - Malware Prediction
  - [Competition overview](https://www.kaggle.com/c/microsoft-malware-prediction)
  - [Winner (2nd) blog/article](https://www.kaggle.com/c/microsoft-malware-prediction/discussion/84065#latest-503765)
  - [Winner (2nd) notebook/code/kernel](https://github.com/imor-de/microsoft_malware_prediction_kaggle_2nd)
  - Other notebook/code/kernel - NA
  - Take home message - NA  
  
## 217 VSB Power Line Fault Detection
*Can you detect faults in above-ground electrical lines?*
  - Type - Faults Detection
  - [Competition overview](https://www.kaggle.com/c/vsb-power-line-fault-detection)
  - [Winner (2nd) blog/article](https://www.kaggle.com/c/vsb-power-line-fault-detection/discussion/86616#latest-501584)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  
  
## 218 Histopathologic Cancer Detection
*Identify metastatic tissue in histopathologic scans of lymph node sections*
  - Type - Image Classification
  - [Competition overview](https://www.kaggle.com/c/histopathologic-cancer-detection)
  - [Winner (17th) blog/article](https://www.kaggle.com/c/histopathologic-cancer-detection/discussion/87397#latest-515386)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 219 Google Cloud & NCAA® ML Competition 2019-Women's
*Apply Machine Learning to NCAA® March Madness®*
  - Type - 
  - [Competition overview]()
  - [Winner blog/article](https://www.kaggle.com/c/womens-machine-learning-competition-2019/discussion/88451#latest-514486)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 220 Google Cloud & NCAA® ML Competition 2019-Men's
Apply Machine Learning to NCAA® March Madness*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/mens-machine-learning-competition-2019)
  - [Winner blog/article](https://www.kaggle.com/c/mens-machine-learning-competition-2019/discussion/89150#latest-516634)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 221 PetFinder.my Adoption Prediction
*How cute is that doggy in the shelter?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/petfinder-adoption-prediction)
  - [Winner (2nd) blog/article](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/88773#latest-515044)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 222 Santander Customer Transaction Prediction
*Can you identify who will make a transaction?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/santander-customer-transaction-prediction)
  - [Winner blog/article](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/89003#latest-521279)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA   

## 223 CareerCon 2019 - Help Navigate Robots
*Compete to get your resume in front of our sponsors*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/career-con-2019)
  - [Winner (3rd) blog/article](https://www.kaggle.com/c/career-con-2019/discussion/89181#latest-523612)
  - [Winner (3rd) notebook/code/kernel](https://www.kaggle.com/prith189/starter-code-for-3rd-place-solution)
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 224 Gendered Pronoun Resolution
*Pair pronouns to their correct entities*
  - Type - NLP
  - [Competition overview]()
  - [Winner blog/article](https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/90392#latest-544753)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 225 xx Don't Overfit!
*A Fistful of Samples*
  - Type -
  - [Competition overview](https://www.kaggle.com/c/dont-overfit-ii)
  - [Winner blog/article](https://www.kaggle.com/c/dont-overfit-ii/discussion/91766)
  - [Winner notebook/code/kernel](https://www.kaggle.com/zachmayer/first-place-solution?scriptVersionId=13934694)
  - [Other notebook/code/kernel](https://medium.com/analytics-vidhya/kaggle-competition-dont-overfit-ii-74cf2d9deed5)
  - Take home message: The critical insight was being among the first to realize leaderboard probing is the key to winning.

## 226 Google Landmark Recognition 2019
*Label famous (and not-so-famous) landmarks in images*
  - Type - Image Labelling
  - [Competition overview](https://www.kaggle.com/c/landmark-recognition-2019)
  - [Winner (2nd) blog/article](https://arxiv.org/pdf/1906.03990.pdf)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 227 LANL Earthquake Prediction
*Can you predict upcoming laboratory earthquakes?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/LANL-Earthquake-Prediction)
  - [Winner blog/article](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94390#latest-568449)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 228 Google Landmark Retrieval 2019
*Given an image, can you find all of the same landmarks in a dataset?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/landmark-retrieval-2019)
  - [Winner blog/article](https://arxiv.org/pdf/1906.04087.pdf)
  - [Winner notebook/code/kernel](https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution)
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 229 iWildCam 2019 - FGVC6
*Categorize animals in the wild*
  - Type - Images Classification
  - [Competition overview](https://www.kaggle.com/c/iwildcam-2019-fgvc6)
  - Winner blog/article - NA
  - [Winner notebook/code/kernel](https://github.com/HayderYousif/iwildcam-2019-fgvc6)
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 230 iMet Collection 2019 - FGVC6
*Recognize artwork attributes from The Metropolitan Museum of Art*
  - Type - Image Classification
  - [Competition overview](https://www.kaggle.com/c/imet-2019-fgvc6)
  - [Winner blog/article](https://www.kaggle.com/c/imet-2019-fgvc6/discussion/94687#latest-570986)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 231 iMaterialist (Fashion) 2019 at FGVC6
*Fine-grained segmentation task for fashion and apparel*
  - Type - Images Segmentation
  - [Competition overview](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6)
  - [Winner blog/article](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/discussion/95247#latest-567841)
  - [Winner notebook/code/kernel](https://github.com/amirassov/kaggle-imaterialist)
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 232 Freesound Audio Tagging 2019
*Automatically recognize sounds and apply tags of varying natures*
  - Type - Audio Tagging 
  - [Competition overview](https://www.kaggle.com/c/freesound-audio-tagging-2019)
  - [Winner blog/article](https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/95924)
  - [Winner notebook/code/kernel](https://github.com/lRomul/argus-freesound)
  - Other notebook/code/kernel - NA
  - Take home message - NA   

## 233 Instant Gratification
*A synchronous Kernels-only competition*
  - Type
  - [Competition overview](https://www.kaggle.com/c/instant-gratification/discussion/96549#latest-564990)
  - [Winner blog/article](https://www.kaggle.com/c/instant-gratification/discussion/96549#latest-564990)
  - [Winner notebook/code/kernel](https://www.kaggle.com/infinite/v2-all-gmm)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 234 Jigsaw Unintended Bias in Toxicity Classification
*Detect toxicity across a diverse range of conversations*
  - Type - NLP
  - [Competition overview](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
  - [Winner blog/article](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/103280#latest-619135)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 235 Northeastern SMILE Lab - Recognizing Faces in the Wild
*Can you determine if two individuals are related?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/recognizing-faces-in-the-wild)
  - [Winner blog/article](https://www.kaggle.com/c/recognizing-faces-in-the-wild/discussion/103670#latest-605952)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 236 Generative Dog Images
*Experiment with creating puppy pics*
  - Type - Generative Images
  - [Competition overview](https://www.kaggle.com/c/generative-dog-images)
  - [Winner blog/article](https://www.kaggle.com/c/generative-dog-images/discussion/106324#latest-633231)
  - [Winner notebook/code/kernel](https://www.kaggle.com/tikutiku/gan-dogs-starter-biggan)
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 237 Predicting Molecular Properties
*Can you measure the magnetic interactions between a pair of atoms?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/champs-scalar-coupling)
  - [Winner blog/article](https://www.kaggle.com/c/champs-scalar-coupling/discussion/106575#latest-635305)
  - [Winner notebook/code/kernel](https://github.com/boschresearch/BCAI_kaggle_CHAMPS)
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 238 SIIM-ACR Pneumothorax Segmentation
*Identify Pneumothorax disease in chest x-rays*
  - Type - Vision
  - [Competition overview](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)
  - [Winner blog/article](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107824)
  - [Winner notebook/code/kernel](https://github.com/sneddy/pneumothorax-segmentation)
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 239 APTOS 2019 Blindness Detection
*Detect diabetic retinopathy to stop blindness before it's too late *
  - Type - Vision
  - [Competition overview](https://www.kaggle.com/c/aptos2019-blindness-detection)
  - [Winner blog/article](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 240 Recursion Cellular Image Classification
*CellSignal: Disentangling biological signal from experimental noise in cellular images*
  - Type - Vision
  - [Competition overview](https://www.kaggle.com/c/recursion-cellular-image-classification)
  - [Winner blog/article](https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110543)
  - [Winner notebook/code/kernel](https://github.com/maciej-sypetkowski/kaggle-rcic-1st)
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 241 Open Images 2019 - Visual Relationship
*Detect pairs of objects in particular relationships*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/open-images-2019-visual-relationship)
  - [Winner (2nd) blog/article](https://www.kaggle.com/c/open-images-2019-visual-relationship/discussion/111361)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA   
  
## 242 Open Images 2019 - Object Detection
*Detect objects in varied and complex images*
  - Type - Vision
  - [Competition overview](https://www.kaggle.com/c/open-images-2019-object-detection)
  - [Winner (6th) blog/article](https://www.kaggle.com/c/open-images-2019-object-detection/discussion/110953)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 243 Open Images 2019 - Instance Segmentation
*Outline segmentation masks of objects in images*
  - Type - Vision
  - [Competition overview](https://www.kaggle.com/c/open-images-2019-instance-segmentation)
  - [Winner (7th) blog/article](https://www.kaggle.com/c/open-images-2019-instance-segmentation/discussion/110983)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 244 IEEE-CIS Fraud Detection
*Can you detect fraud from customer transactions?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/ieee-fraud-detection)
  - [Winner blog/article](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 245 The 3rd YouTube-8M Video Understanding Challenge
*Temporal localization of topics within video*
  - Type - Vision
  - [Competition overview](https://www.kaggle.com/c/youtube8m-2019)
  - [Winner blog/article](https://www.kaggle.com/c/youtube8m-2019/discussion/112869)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA 
 
## 246 Kuzushiji Recognition
*Opening the door to a thousand years of Japanese culture*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/kuzushiji-recognition)
  - [Winner blog/article](https://www.kaggle.com/c/kuzushiji-recognition/discussion/112788)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 247 Severstal: Steel Defect Detection
*Can you detect and classify defects in steel?*
  - Type - Vision
  - [Competition overview](https://www.kaggle.com/c/severstal-steel-defect-detection)
  - [Winner blog/article](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114254)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA 

## 248 Lyft 3D Object Detection for Autonomous Vehicles
*Can you advance the state of the art in 3D object detection?*
  - Type - Vision
  - [Competition overview](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles)
  - [Winner blog/article](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/discussion/122820)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 249 RSNA Intracranial Hemorrhage Detection
*Identify acute intracranial hemorrhage and its subtypes*
  - Type - Vision
  - [Competition overview](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)
  - [Winner blog/article](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117210)
  - [Winner notebook/code/kernel](https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection)
  - Other notebook/code/kernel - NA
  - Take home message - NA   
  
## 250 Understanding Clouds from Satellite Images
*Can you classify cloud structures from satellites?*
  - Type - Visiion
  - [Competition overview]()
  - [Winner blog/article](https://www.kaggle.com/c/understanding_cloud_organization)
  - [Winner notebook/code/kernel](https://github.com/pudae/kaggle-understanding-clouds)
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 251 2019 Kaggle Machine Learning & Data Science Survey
*The most comprehensive dataset available on the state of ML and data science*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/kaggle-survey-2019)
  - [Winner blog/article](https://www.kaggle.com/tkubacka/a-story-told-through-a-heatmap)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 252 Categorical Feature Encoding Challenge
*Binary classification, with every feature a categorical*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/cat-in-the-dat)
  - [Winner blog/article](https://www.kaggle.com/c/cat-in-the-dat/discussion/121356)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 253 BigQuery-Geotab Intersection Congestion
*Can you predict wait times at major city intersections?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/bigquery-geotab-intersection-congestion)
  - Winner blog/article - NA
  - [Winner notebook/code/kernel](https://github.com/peterhurford/kaggle-intersection)
  - Other notebook/code/kernel - NA
  - Take home message - NA  
  
## 254 Kannada MNIST
*MNIST like datatset for Kannada handwritten digits*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/Kannada-MNIST
  - [Winner blog/article](https://www.kaggle.com/c/Kannada-MNIST/discussion/122230)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 255 ASHRAE - Great Energy Predictor III
*How much energy will a building consume?*
  - Type - Forecasting
  - [Competition overview](https://www.kaggle.com/c/ashrae-energy-prediction)
  - [Winner blog/article](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/124709)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 256 NFL Big Data Bowl
*How many yards will an NFL player gain after receiving a handoff?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/nfl-big-data-bowl-2020)
  - [Winner blog/article](https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119400)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 257 Santa's Workshop Tour 2019
*In the notebook we can build a model, and pretend that it will optimize...*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/santa-workshop-tour-2019)
  - [Winner blog/article](https://www.kaggle.com/c/santa-workshop-tour-2019/discussion/127427)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 258 Santa 2019 - Revenge of the Accountants
*Oh what fun it is to revise . . .*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/santa-2019-revenge-of-the-accountants)
  - [Winner blog/article](https://www.kaggle.com/c/santa-2019-revenge-of-the-accountants/discussion/126380)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  
  
## 259 Peking University/Baidu - Autonomous Driving
*Can you predict vehicle angle in different settings?*
  - Type - Vision
  - [Competition overview](https://www.kaggle.com/c/pku-autonomous-driving)
  - [Winner blog/article](https://www.kaggle.com/c/pku-autonomous-driving/discussion/127037)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 260 2019 Data Science Bowl
*Uncover the factors to help measure how young children learn*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/data-science-bowl-2019)
  - [Winner blog/article](https://www.kaggle.com/c/data-science-bowl-2019/discussion/127469)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
 
## 261 TensorFlow 2.0 Question Answering
*Identify the answers to real user questions about Wikipedia page content*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/tensorflow2-question-answering)
  - [Winner blog/article](https://www.kaggle.com/c/tensorflow2-question-answering/discussion/127551)
  - [Winner notebook/code/kernel](https://github.com/google-research-datasets/natural-questions)
  - Other notebook/code/kernel - NA
  - Take home message - NA
 
## 262 Google QUEST Q&A Labeling
*Improving automated understanding of complex question answer content*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/google-quest-challenge)
  - [Winner blog/article](https://www.kaggle.com/c/google-quest-challenge/discussion/129840)
  - [Winner notebook/code/kernel](https://github.com/oleg-yaroshevskiy/quest_qa_labeling)
  - Other notebook/code/kernel - NA
  - Take home message - NA
 
## 263 Bengali.AI Handwritten Grapheme Classification
*Classify the components of handwritten Bengali*
  - Type - Vision | Classification | Handewriting
  - [Competition overview](https://www.kaggle.com/c/bengaliai-cv19)
  - [Winner blog/article](https://www.kaggle.com/c/bengaliai-cv19/discussion/135984)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
 
## 264 DS4G - Environmental Insights Explorer
*Exploring alternatives for emissions factor calculations*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/ds4g-environmental-insights-explorer)
  - [Winner blog/article](https://www.kaggle.com/katemelianova/ds4g-spatial-panel-data-modeling)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
 
## 265 Categorical Feature Encoding Challenge II
*Binary classification, with every feature a categorical (and interactions!)*
  - Type - Binary Classification
  - [Competition overview](https://www.kaggle.com/c/cat-in-the-dat-ii)
  - [Winner blog/article](https://www.kaggle.com/c/cat-in-the-dat-ii/discussion/140465)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
 
## 266 Deepfake Detection Challenge
*Identify videos with facial or voice manipulations*
  - Type - Video
  - [Competition overview](https://www.kaggle.com/c/deepfake-detection-challenge
  - [Winner blog/article](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145721)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 267 Google Cloud & NCAA® March Madness Analytics
*Uncover the madness of March Madness®*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/march-madness-analytics-2020)
  - [Winner blog/article](https://www.kaggle.com/c/march-madness-analytics-2020/discussion/152213)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
 
## 268 COVID19 Global Forecasting (Week 5)
*Forecast daily COVID-19 spread in regions around world*
  - Type - Forecasting
  - [Competition overview](https://www.kaggle.com/c/covid19-global-forecasting-week-5)
  - [Winner blog/article](https://www.kaggle.com/c/covid19-global-forecasting-week-5/discussion/155638)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  
  
## 269 University of Liverpool - Ion Switching
*Identify the number of channels open at each time point*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/liverpool-ion-switching)
  - [Winner blog/article](https://www.kaggle.com/c/liverpool-ion-switching/discussion/153940)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  

## 270 Herbarium 2020 - FGVC7
*Identify plant species from herbarium specimens. Data from New York Botanical Garden.*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/herbarium-2020-fgvc7)
  - [Winner blog/article](https://www.kaggle.com/c/herbarium-2020-fgvc7/discussion/154351)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 271 iWildCam 2020 - FGVC7
*Categorize animals in the wild*
  - Type - Classification
  - [Competition overview](https://www.kaggle.com/c/iwildcam-2020-fgvc7)
  - [Winner blog/article](https://www.kaggle.com/c/iwildcam-2020-fgvc7/discussion/158370)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 272 iMaterialist (Fashion) 2020 at FGVC7
*Fine-grained segmentation task for fashion and apparel*
  - Type - Classification | Images Segmentation
  - [Competition overview](https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7)
  - [Winner blog/article](https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7/discussion/154306)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 273 Plant Pathology 2020 - FGVC7
*Identify the category of foliar diseases in apple trees*
  - Type - Vision | Classification
  - [Competition overview](https://www.kaggle.com/c/plant-pathology-2020-fgvc7)
  - [Winner blog/article](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/discussion/154056)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 274 Abstraction and Reasoning Challenge
*Create an AI capable of solving reasoning tasks it has never seen before*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/abstraction-and-reasoning-challenge)
  - [Winner blog/article](https://www.kaggle.com/c/abstraction-and-reasoning-challenge/discussion/154597)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
 
## 275 Tweet Sentiment Extraction
*Extract support phrases for sentiment labels*
  - Type - NLP | Sentiment Analysis
  - [Competition overview](https://www.kaggle.com/c/tweet-sentiment-extraction)
  - [Winner blog/article](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159477)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
 
## 276 Jigsaw Multilingual Toxic Comment Classification
*Use TPUs to identify toxicity comments across multiple languages*
  - Type - NKP | Classification | Sentiment Analysis
  - [Competition overview](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification)
  - [Winner blog/article](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160862)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA

## 277 TReNDS Neuroimaging
*Multiscanner normative age and assessments prediction with brain function, structure, and connectivity*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/trends-assessment-prediction)
  - [Winner blog/article](https://www.kaggle.com/c/trends-assessment-prediction/discussion/163017)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA  
  - Take home message - NA  
  
## 278 M5 Forecasting - Uncertainty
*Estimate the uncertainty distribution of Walmart unit sales.*
  - Type - Forecasting
  - [Competition overview](https://www.kaggle.com/c/m5-forecasting-uncertainty)
  - [Winner blog/article](https://www.kaggle.com/c/m5-forecasting-uncertainty/discussion/163368)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 279 M5 Forecasting - Accuracy
*Estimate the unit sales of Walmart retail goods*
  - Type - Forecasting
  - [Competition overview](https://www.kaggle.com/c/m5-forecasting-accuracy)
  - [Winner blog/article](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163684)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 280 ALASKA2 Image Steganalysis
*Detect secret data hidden within digital images*
  - Type - Vision
  - [Competition overview](https://www.kaggle.com/c/alaska2-image-steganalysis)
  - [Winner blog/article](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/168548)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA 
  - Take home message - NA
  
## 281 Prostate cANcer graDe Assessment (PANDA) Challenge
*Prostate cancer diagnosis using the Gleason grading system*
  - Type - Medicine | Vision
  - [Competition overview](https://www.kaggle.com/c/prostate-cancer-grade-assessment)
  - Winner blog/article - NA
  - [Winner notebook/code/kernel](https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution)
  - Other notebook/code/kernel - NA  
  - Take home message - NA
  
## 282 Hash Code Archive - Photo Slideshow Optimization
*Optimizing a photo album from Hash Code 2019*
  - Type - Vision
  - [Competition overview](https://www.kaggle.com/c/hashcode-photo-slideshow)
  - [Winner blog/article](https://www.kaggle.com/c/hashcode-photo-slideshow/discussion/170575)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA  
  - Take home message - NA
  
## 283 SIIM-ISIC Melanoma Classification
*Identify melanoma in lesion images*
  - Type - Vision | Medicine
  - [Competition overview](https://www.kaggle.com/c/siim-isic-melanoma-classification)
  - [Winner blog/article](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175412)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA  
  - Take home message - NA
  
## 284 Google Landmark Retrieval 2020
*Given an image, can you find all of the same landmarks in a dataset?*
  - Type - Vision 
  - [Competition overview](https://www.kaggle.com/c/landmark-retrieval-2020)
  - [Winner blog/article](https://arxiv.org/ftp/arxiv/papers/2009/2009.05132.pdf)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA  
  - Take home message - NA
  
## 285 Global Wheat Detection
*Can you help identify wheat heads using image analysis?*
  - Type - Vision
  - [Competition overview](https://www.kaggle.com/c/global-wheat-detection)
  - [Winner blog/article](https://www.kaggle.com/c/global-wheat-detection/discussion/172418)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA  
  - Take home message - NA
  
## 286 Cornell Birdcall Identification
*Build tools for bird population monitoring*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/birdsong-recognition)
  - [Winner blog/article](https://www.kaggle.com/c/birdsong-recognition/discussion/183208)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA  
  - Take home message - NA
  
## 287 Halite by Two Sigma
*Collect the most halite during your match in space*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/halite)
  - [Winner blog/article](https://www.kaggle.com/c/halite/discussion/183543)
  - [Winner notebook/code/kernel](https://github.com/ttvand/Halite)
  - Other notebook/code/kernel - NA  
  - Take home message - NA
  
## 288 Google Landmark Recognition 2020
*Label famous (and not-so-famous) landmarks in images*
  - Type - Vision
  - [Competition overview](https://www.kaggle.com/c/landmark-recognition-2020)
  - [Winner blog/article](https://arxiv.org/pdf/2010.01650.pdf)
  - [Winner notebook/code/kernel](https://github.com/psinger/kaggle-landmark-recognition-2020-1st-place)
  - Other notebook/code/kernel - NA  
  - Take home message - NA
  
## 289 OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction
*Urgent need to bring the COVID-19 vaccine to mass production*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/stanford-covid-vaccine)
  - [Winner blog/article](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189620)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA 
  - Take home message - NA
  
## 290 OSIC Pulmonary Fibrosis Progression
*Predict lung function decline*
  - Type - Medicine
  - [Competition overview](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression)
  - [Winner blog/article](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/discussion/189346)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA     
  - Take home message - NA
  
## 291 RSNA STR Pulmonary Embolism Detection
*Classify Pulmonary Embolism cases in chest CT scans*
  - Type - Medicine | Vision
  - [Competition overview](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection)
  - [Winner blog/article](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/194145)
  - [Winner notebook/code/kernel](https://github.com/GuanshuoXu/RSNA-STR-Pulmonary-Embolism-Detection)
  - Other notebook/code/kernel - NA
  - Take home message - Since the provided data are big and of high quality, **we don't have to do cross validation**, a single training/validation split is reliable enough.

## 292 Lyft Motion Prediction for Autonomous Vehicles
*Build motion prediction models for self-driving vehicles*
  - Type - Vision | Autonomous Vehicles
  - [Competition overview](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles)
  - [Winner blog/article](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/201493)
  - [Winner (3rd) blog/article](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/205376)
  - [Winner (3rd) notebook/code/kernel](https://github.com/asanakoy/kaggle-lyft-motion-prediction-av)
  - Other notebook/code/kernel - NA
  - Take home message - NA  
  
## 293 Conway's Reverse Game of Life 2020
*Reverse the arrow of time in the Game of Life*
  - Type - NA
  - [Competition overview](https://www.kaggle.com/c/conways-reverse-game-of-life-2020)
  - [Winner blog/article](https://www.kaggle.com/c/conways-reverse-game-of-life-2020/discussion/200980)
  - [Winner (3rd) blog/article](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/205376)
  - [Winner (3rd) notebook/code/kernel](https://github.com/asanakoy/kaggle-lyft-motion-prediction-av)
  - Other notebook/code/kernel - NA
  - Take home message - NA
  
## 294 Mechanisms of Action (MoA) Prediction
*Can you improve the algorithm that classifies drugs based on their biological activity?*
  - Type - Classification
  - [Competition overview](https://www.kaggle.com/c/lish-moa)
  - [Winner blog/article](https://www.kaggle.com/c/lish-moa/discussion/201510)  
  - [Winner notebook/code/kernel](https://github.com/guitarmind/kaggle_moa_winner_hungry_for_gold)
  - Other notebook/code/kernel - NA
  - Take home message - NA  
  
## 295 Google Research Football with Manchester City F.C.
*Train agents to master the world's most popular sport*
  - Type - Reinforced Learning
  - [Competition overview](https://www.kaggle.com/c/google-football)
  - [Winner blog/article](https://www.kaggle.com/c/google-football/discussion/202232)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA 
  
## 296 Hash Code Archive - Drone Delivery
*Can you help coordinate the drone delivery supply chain?*
  - Type - 
  - [Competition overview](https://www.kaggle.com/c/hashcode-drone-delivery)
  - [Winner blog/article](https://www.kaggle.com/c/hashcode-drone-delivery/discussion/204876)
  - Winner notebook/code/kernel - NA
  - Other notebook/code/kernel - NA
  - Take home message - NA  
