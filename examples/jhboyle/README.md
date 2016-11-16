# Classifying 1984 US House of Representatives Voting Records by Party Affiliation Information

Example case study of binary-classification, using party-affiliation as the target feature,
for the 1984 US House of Representatives' voting records. 

Python code includes data fetching, exploration and visualization, and Logistic Regression
classification using GridSearchCV.

Downloaded from the UCI Machine Learning Repository on 13 November 2016. The dataset description is as follows:

    Data Set: Multivariate
    Attribute: Real
    Tasks: Classification
    Instances: 435
    Attributes: 16
    Missing Values: Yes
    Area: Social
    Date Donated: 1987-04-27

## Data Set Information:

This data set includes votes for each of the U.S. House of Representatives Congressmen on the 16 key votes identified by the CQA. The CQA lists nine different types of votes: voted for, paired for, and announced for (these three simplified to yea), voted against, paired against, and announced against (these three simplified to nay), voted present, voted present to avoid conflict of interest, and did not vote or otherwise make a position known (these three simplified to an unknown disposition).
Attribute Information:

    Class Name: 2 (democrat, republican)
    handicapped-infants: 2 (y,n)
    water-project-cost-sharing: 2 (y,n)
    adoption-of-the-budget-resolution: 2 (y,n)
    physician-fee-freeze: 2 (y,n)
    el-salvador-aid: 2 (y,n)
    religious-groups-in-schools: 2 (y,n)
    anti-satellite-test-ban: 2 (y,n)
    aid-to-nicaraguan-contras: 2 (y,n)
    mx-missile: 2 (y,n)
    immigration: 2 (y,n)
    synfuels-corporation-cutback: 2 (y,n)
    education-spending: 2 (y,n)
    superfund-right-to-sue: 2 (y,n)
    crime: 2 (y,n)
    duty-free-exports: 2 (y,n)
    export-administration-act-south-africa: 2 (y,n)

## Relevant Papers:

Schlimmer, J. C. (1987). Concept acquisition through representational adjustment. Doctoral dissertation, Department of Information and Computer Science, University of California, Irvine, CA.

