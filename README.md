# Disaster Response Pipeline Project

This pipeline is an app that classifies messages in order for help to be arranged more efficiently.

Typically after a natural disaster, the government may receive many information (such as tweets) from people who needs help, and since the matters are urgent, it is very important to develop a tool that classify the messages correctly and efficiently so the help can be arranged on time.

This web app is exactly achieving this purpose.

Once you open the web in your browser (the instruction of how to open the app is down below), you could input a message in the searching box, then the app will classify what type of message it is (for example, aid related, medical related, and/or request). According the classfied output, the help will be arranged more efficiently. 


### Instructions for using the app:
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

Run the following command in the app's directory to run your web app.
    `python run.py`

Go to https://view6914b2f4-3001.udacity-student-workspaces.com/
