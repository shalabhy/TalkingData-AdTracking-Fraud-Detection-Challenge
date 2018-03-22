# TalkingData-AdTracking-Fraud-Detection-Challenge
aim - to build an algorithm that predicts whether a user will download an app after clicking a mobile app ad.
sol - The file size was very large, training data has about 200 million rows. The data is highly unbalanced as only about 0.2% times the clicks have actually been converted into downloads.
The solution begins with importing the data in chunks so that the preprocessing can be efficient. After this  from all the chunks
rows in which click is turned to download are selcted and two times the rows for which there is no download are selected merged with other data and shuffled. This way the final data obtained has 1/3rd data with is_attributed = 1. After this separate columns for date and day are made and light gbm model is applied for prediction. 


