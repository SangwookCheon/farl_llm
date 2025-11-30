from main import get_llm_recommendation

info_dict = {
    'gender':1.0,'age':56.0,'race':2.0,'educ':3.0,'marry':1.0,
    'house':1.0,'pov':2.15,'wt':96.4,'ht':168.0,'bmi':34.1,
    'wst':110.5,'hip':112.0,'dia':84.0,'pulse':78.0,'sys':138.0,
    'alt':42.0,'albumin':4.3,'ast':35.0,'crea':0.98,'chol':205.0,
    'tyg':230.0,'ggt':48.0,'wbc':7.2,'hb':14.2,'hct':43.0,
    'ldl':132.0,'hdl':38.0,'acratio':3.1,'glu':182.0,
    'insulin':18.5,'crp':4.8,'hb1ac':8.6,'mvpa':40.0,
    'ac_week':0.0,
    'context':'Middle-aged patient with long-standing type 2 diabetes, obesity, inconsistent follow-up, and limited financial resources. Frequently eats late meals due to shift-based work schedule, with high intake of refined carbohydrates and sugary beverages. Reports chronic knee discomfort limiting high-impact exercise. Demonstrates motivation to prevent future complications but experiences difficulty implementing recommendations without clear, structured steps.',
    "1week":("mvpa",20.0),
    "2week":("ac_week",1.0),
    "3week":("sugary_drinks_per_day",-0.5),
    "4week":("wst",-1.0),
    "5week":("glu",-5.0),
    "6week":("hb1ac",-0.1),
    "7week":("tyg",-10.0),
    "8week":("sys",-2.0),
    "old_score":103.0,
    "new_score":93.0
}

text = get_llm_recommendation(info_dict)
print(text)

