# default values
seed = 100
headers = {
    "adult": ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class-salary"],
    "student": ["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","guardian","traveltime","studytime","failures","schoolsup","famsup","paid","activities","nursery","higher","internet","romantic","famrel","freetime","goout","Dalc","Walc","health","absences","G1","G2","G3"],
    #"athletes": ["ID","Name","Sex","Age","Height","Weight","Team","NOC","Games","Year","Season","City","Sport","Event","Medal"],
    "bank": ["age","job","marital","education","default","housing","loan","contact","month","day_of_week","duration","campaign","pdays","previous","poutcome","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed","y"],
    "creditcardcustomers": ["Attrition_Flag","Customer_Age","Gender","Dependent_count","Education_Level","Marital_Status","Income_Category","Card_Category","Months_on_book","Total_Relationship_Count","Months_Inactive_12_mon","Contacts_Count_12_mon","Credit_Limit","Total_Revolving_Bal","Avg_Open_To_Buy","Total_Amt_Chng_Q4_Q1","Total_Trans_Amt","Total_Trans_Ct","Total_Ct_Chng_Q4_Q1","Avg_Utilization_Ratio"]
}

NOT_SENSITIVE = 0
SENSITIVE = 1

# each entry denotes two subsets: the first subset refers to non-sensitive attributes, the second the sensitive attributes
initial_subsets = {
    "adult": [{"age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "class-salary", "hours-per-week", "workclass", "occupation"}, {"race", "sex", "native-country", "relationship", "marital-status", "education"} ],
    "student": [{"school","age","address","famsize","Pstatus","reason","guardian","traveltime","studytime","failures","schoolsup","famsup","paid","activities","nursery","higher","internet","romantic","famrel","freetime","goout","Dalc","Walc","health","absences","G1","G2","G3"},{"sex", "Medu","Fedu","Mjob","Fjob"}],
    #"athletes": [{"ID","Name","Sex","Age","Height","Weight","Team","NOC","Games","Year","Season","City","Sport","Event","Medal"}, {}],
    "bank": [{"age","housing","loan","contact","month","day_of_week","duration","campaign","pdays","previous","poutcome","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed","y", "default"}, {"marital", "job", "education"}],
    "creditcardcustomers": [{"Attrition_Flag","Customer_Age","Dependent_count","Income_Category","Card_Category","Months_on_book","Total_Relationship_Count","Months_Inactive_12_mon","Contacts_Count_12_mon","Credit_Limit","Total_Revolving_Bal","Avg_Open_To_Buy","Total_Amt_Chng_Q4_Q1","Total_Trans_Amt","Total_Trans_Ct","Total_Ct_Chng_Q4_Q1","Avg_Utilization_Ratio"}, {"Gender", "Marital_Status","Education_Level"}]
}

nulls = {
    "adult": " ?",
    "bank": "unknown",
    "creditcardcustomers": "Unknown",
    "student": None
}