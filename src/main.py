import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from taipy import Gui

# sys.path.insert(0, os.path.abspath(".."))
# sys.path.append('../src')

# Import Main Libraries
from models import Preprocessor

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Library for expand comments function.
import nltk

# Set options and formatting
pd.set_option("display.precision", 1)
# from matplotlib_inline.backend_inline import set_matplotlib_formats
# set_matplotlib_formats('svg')
# %config InlineBackend.figure_format = 'svg'

from models.DeepLearning import get_dominant_topic, get_lda_topics, get_nmf_topics, get_extractive_summaries, get_t5_summary, get_lda_coherence_chart, visualise_lda

# GUI
options_piechart_thematic = [
    # First pie chart
    {
        # Show label value on hover
        "hoverinfo": "label",
        # Leave a hole in the middle of the chart
        "hole": 0.4,
        # Place the trace on the left side
        "domain": {"column": 0}
    },
    # Second pie chart
    {
        # Show label value on hover
        "hoverinfo": "label",
        # Leave a hole in the middle of the chart
        "hole": 0.4,
        # Place the trace on the right side
        "domain": {"column": 1}
    }
]

layout_longitudinal = {
    # Hide the legend
    "showlegend": False,
    "title": "% of LDA tokens allocated to this theme every year"
}

layout_piechart_thematic = {
    # Chart title
    "title": "Percentage of reviews (LDA tokens) allocated to each topic in the NSS Data for The Statistics Department",
    # Show traces in a 1x2 grid
    "grid": {
        "rows": 1,
        "columns": 2
    },
    "annotations": [
        # Annotation for the first trace
        {
            "text": "Positive",
            "font": {
                "size": 18
            },
            # Hide annotation arrow
            "showarrow": False,
            # Move to the center of the trace
            "x": 0.202,
            "y": 0.5
        },
        # Annotation for the second trace
        {
            "text": "Negative",
            "font": {
                "size": 18
            },
            "showarrow": False,
            # Move to the center of the trace
            "x": 0.805,
            "y": 0.5
        }
    ],
    "showlegend": True
}

df_community_department_positive = pd.DataFrame({"Year":[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
                          "Community":[0,0,0,0,15.5,0,37.4,0,13.9],
                          "Department":[0,7.6,0,0,15.5,0,16.6,0,14],
                          "Community & Department":[0,7.6,0,0,31,0,35.5,50.1,36.8]})

df_course_content_positive = pd.DataFrame({"Year":[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
                          "Course & Content":[55.1,92.4,83.1,36.2,68.9,57.8,38.4,15,33.2]})

df_teaching_learning_setup_positive = pd.DataFrame({"Year":[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
                          "Teaching & Learning Set-Up":[44.9,0,16.9,63.8,0,42.2,26.1,34.9,30]})

df_community_department_negative = pd.DataFrame({"Year":[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
                          "Community":[0,12.5,0,0,0,0,37.4,0,13.9],
                          "Department":[0,0,0,24.8,0,12.6,16.6,0,14],
                          "Community & Department":[0,12.5,0,24.8,0,12.6,54,26.2,27.9]})

df_course_content_negative = pd.DataFrame({"Year":[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
                          "Course & Content":[37.3,28.2,72.8,17.9,46.8,46.9,21.1,0,25.3]})

df_teaching_learning_setup_negative = pd.DataFrame({"Year":[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
                          "Assessment & Feedback":[0,0,0,0,0,0,0,28.8,15.6],
                          "Teaching & Exams":[0,21,0,13.7,17.25,0,0,28,0],
                          "Learning Set-Up":[0,38.3,27.3,43.7,35.95,0,0,17,31.2],
                          "Teaching & Learning Set-Up":[32.5,59.3,27.3,57.4,53.2,40.4,25,73.8,46.8]})


# Diagnosis 2014/15 Table
diagnosis_2015_positive = pd.DataFrame(columns=['Theme', 'Percentage of LDA Tokens', 'Strength'])
diagnosis_2015_positive.loc[0, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Content & Modules', '55.1%',
        '''
        flexible module choices/options, dissertation support, interesting content, range of modules, 3rd year modules interesting.
        ''']

diagnosis_2015_positive.loc[1, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Teaching', '44.9%',
        '''
        good teaching, well run department, dedicated, knowledgeable, and passionate teaching, quality of probability modules. 
        ''']

diagnosis_2015_negative = pd.DataFrame(columns=['Theme', 'Percentage of LDA Tokens', 'Complaint'])
diagnosis_2015_negative.loc[0, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Content', '37.3%',
    '''
    module tailored to careers in statistics, not enough motivation/applicability/examples provided, insufficient learning resources to prepare for exams., typesetted notes for level 3 and 4 modules, first year hard and adaptability issues.
    ''']
diagnosis_2015_negative.loc[1, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Teaching & Learning Set-Up', '32.5%',
    '''
    lack of group work and collaboration opportunities, negative teaching experience, unclear about what is examinable, inconsistent standard of supervisors, inter-departmental communication, poor teaching
    notes not available online, take 9 modules exams in 1 sitting, recorded lectures not accessible for exam revision, lecturers copy from pages so boring lectures, description of module motivation so that you don't realise importance of everything at the end of your degree.
    ''']


# Diagnosis 2015/16 Table
diagnosis_2016_positive = pd.DataFrame(columns=['Theme', 'Percentage of LDA Tokens', 'Strength'])
diagnosis_2016_positive.loc[0, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Modules', '35.2%',
        '''
        Flexibility of options, range/breadth of modules
        variety, choice, explore interests, challenging, inter-departmental options.
        ''']

diagnosis_2016_positive.loc[1, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Content', '18.2%',
        '''
        difficult, rewarding,e employable, stimulating, apply math knowledge, wide range of modules makes it easy to specialise.
        ''']

diagnosis_2016_positive.loc[2, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Course', '39%',
        '''
        Support classes especially first and second year, breadth and depth of options, support facilities, regular communication, Diversity
        good lectures, quick lecture email response, good teaching methodologies, good layout of courses, interesting and useful content, allows you to explore your interests.
        specialise in third year, friendly and helpful support office, employable course, well-organised, enjoyable, modules run efficiently and good learning support with problem classes, supervisions and assignments. 
        ''']

diagnosis_2016_positive.loc[3, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Department', '7.6%',
        '''
        intellectually stimulating course, communication with department, online notes, optional modules choices.
        ''']

diagnosis_2016_negative = pd.DataFrame(columns=['Theme', 'Percentage of LDA Tokens', 'Complaint'])
diagnosis_2016_negative.loc[0, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'First Year Adaptation & Guidance', '38.3%',
    '''
    module guidance, 10% counting creates pressure for first year students, less able to explore career opportunities, minimal career search in first year, too late career search in final year, adapatability because 10% first year.
    assignments first two weeks cannot socialise, other degrees have more time to look into career opportunities.

    ''']
diagnosis_2016_negative.loc[1, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Teaching', '21%',
    '''
    Lack of resources, no online notes in later years, lack of reading material, inaccessibly lecture recordings, more support for straight math students.
    ''']
diagnosis_2016_negative.loc[2, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Course', '17.8%',
    '''
    difficuly modules, teaching quality, too many financial applications and not others.
    ''']
diagnosis_2016_negative.loc[3, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Community', '12.5%',
    '''
    lack of study space, lack of communication from econ department, lack of involvement in extracurriculars due to timetabling and workload.
    ''']
diagnosis_2016_negative.loc[4, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Content', '10.4%',
    '''
    more emphasis on R and programming skills, lectures go too fast, lack of chance to specialise earlier on in the degree, too many core modules in first two years, more guidance on utility of certain courses.
    ''']

# Diagnosis 2017 Table
diagnosis_2017_positive = pd.DataFrame(columns=['Theme', 'Percentage of LDA Tokens', 'Strength'])
diagnosis_2017_positive.loc[0, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Course', '45.4%',
        '''
        flexible module choices/options, variety, challenging course, teaching, organised, enjoyable, community.
        ''']
diagnosis_2017_positive.loc[1, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Modules', '37.7%',
        '''
        variety, range of options, second year modules, employability, relevance to real-world [bid data, analytics, etc...], 4 departments module options.
        ''']
diagnosis_2017_positive.loc[2, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Learning Set-Up', '16.9%',
        '''
        module choice eligibility, resources and teaching, change personal tutors, quality of modules.
        ''']

diagnosis_2017_negative = pd.DataFrame(columns=['Theme', 'Percentage of LDA Tokens', 'Complaint'])
diagnosis_2017_negative.loc[0, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Course', '43.6%',
    '''
    difficult to click with people, teaching, module guidance, too quick pace, adapting to first year
    ''']
diagnosis_2017_negative.loc[1, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Modules', '29.2%',
    '''
    over-emphasis on theory that is inapplicable and unehelpful for careers, adapting to first year, mentors unhelpful, no past-paper solutions.  
    ''']
diagnosis_2017_negative.loc[2, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Learning Set-Up', '27.3%',
    '''
    lack of support for academic learning, food options in zeeman, no feedback on exam paper, lack of module selection guidance, transition from college to uni.
    ''']

# Diagnosis 2018 Table
diagnosis_2018_positive = pd.DataFrame(columns=['Theme', 'Percentage of LDA Tokens', 'Strength'])
diagnosis_2018_positive.loc[0, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Course', '36.2%',
        '''
        supportive and approachable staff, challenging course, flexibility and adaptable, varied and interesting, exposure to multiple departments maths stats econ, enjoy overlaps in third year, resources and organisation.
        ''']
diagnosis_2018_positive.loc[1, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Teaching', '33.6%',
        '''
        volunteers, career opportunities, quality and content of teaching, helpful lecturers, good lecturers, invested in personal development.
        ''']
diagnosis_2018_positive.loc[2, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Learning Set-Up', '30.2%',
        '''
        clear and engaging teaching, variety of classes develop soft and hard skills and enhance employability, helpful teachers, good course, third year module choices flexibility, ds breadth of choice stats and cs.
        ''']


diagnosis_2018_negative = pd.DataFrame(columns=['Theme', 'Percentage of LDA Tokens', 'Complaint'])
diagnosis_2018_negative.loc[0, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Department', '24.8%',
    '''
    Career guidance, study spaces, food options, opportunity to group work, superficial understanding of concepts given breadth.
    ''']

diagnosis_2018_negative.loc[1, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Course', '17.9%',
    '''
    teaching, more math in first two years, no personal relationship with mentors/supervisors/lecturers/tutors, opportunities to collaborate in first year.
    ''']

diagnosis_2018_negative.loc[2, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Learning Set-Up', '43.7%',
    '''
    small study space, timetabling issue (9 am, then 6pm), study resouces (lecture notes).
    lack of interaction, slow response to emails, timetables rooms far apart, lectures finish at the hour.
    motivation of topic/applicability, no marking scheme, module selection/content guidance and what they lead to.
    ''']

diagnosis_2018_negative.loc[3, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Teaching', '13.7%',
    '''
    inconsistent, poor quality, too one-dimensional, supervisor favouritism, lack of guidance on selecting modules especially in year 1 and 2.
    ''']


# Diagnosis 2019 Table
diagnosis_2019_positive = pd.DataFrame(columns=['Theme', 'Percentage of LDA Tokens', 'Strength'])
diagnosis_2019_positive.loc[0, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Modules', '32.5%',
        '''
        flexibility, range from different departments, applicability to work, support network to study modules tutors, buddies, mentors.    

        choices, combinations, breadth to explore interests.
        ''']

diagnosis_2019_positive.loc[1, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Content', '16.4%',
        '''
        module compliements, opportunity to upskill in maths, stats, and cs. content focuses on personal learning and allows flexible learning approaches, software options and packages (technical skills), applicable to real-world context.
        ''']

diagnosis_2019_positive.loc[2, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Department', '15.5%',
        '''
            academic staff,research, module variety, efficient staff, new 2018/19 building.
        ''']

diagnosis_2019_positive.loc[3, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Community', '15.5%',
        '''
        helpful staff for career and academic advice, resources and feedback, career fairs, attitude towards learning, exposure to different cultures.
        ''']

diagnosis_2019_positive.loc[4, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Course', '20%',
        '''
    workload, teachers provide real-world applications and motivations for statistical techniques, problem-solving skills development, course practicality gives student edge in interviews and ACs.
    
    mentoring scheme, employability, network/course-mates talented/smart, lecturers.
            ''']

diagnosis_2019_negative = pd.DataFrame(columns=['Theme', 'Percentage of LDA Tokens', 'Complaint'])
diagnosis_2019_negative.loc[0, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Teaching & Learning Set-Up', '19.9%',
    '''
    poor course/exam timetable scheduling, lack of lecture recordings, personal tutor disinterest, late release of exam timetable, teaching (good researcher != good teacher forintroductory level), lack of lecture notes and revision guides.
    ''']

diagnosis_2019_negative.loc[1, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Learning Set-Up', '26%',
    '''
    subpar teaching, poor communication between 4 departments, disorganised lecturers, compulsory third year math modules should be optional, too many rules and restrictions on what modules to take.
     
     awful experience dealing with mitigating circumstances and extensions, no access to some more course material before registering for course.
     ''']

diagnosis_2019_negative.loc[2, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Modules', '23%',
    '''
    lecture notes/lecture capture for some modules, board illegible sometimes, lack of guidance when selecting modules.

    lecture rooms size/desk availability, badly taught, lack of notes
    ''']

diagnosis_2019_negative.loc[3, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Content', '23.8%',
    '''
    difficult first year (Adaptability), opportunity costs of taking second year modules don't meet requirements of other third year modules, some courses limited to 4th years.
    
    insufficient exercisesand actionable content, difficult, lack of support during module selection, lack of motivation and applicability of theory.
    ''']

diagnosis_2019_negative.loc[4, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Exam', '7.3%',
    '''
    insufficient mocks/feedback to understand where students have gone wrong, no model exam answers, no motivation/application/usefulness of content explained. 
    ''']

# Diagnosis 2020 Table
diagnosis_2020_positive = pd.DataFrame(columns=['Theme', 'Percentage of LDA Tokens', 'Strength'])
diagnosis_2020_positive.loc[0, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Teaching & Learning Set-Up', '42.2%',
        '''
        set-up variation, supportive supervisor and personal tutor, online notesand teaching, learning R, department support.
    
        ''']

diagnosis_2020_positive.loc[1, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Course', '35.9%',
        '''
       module variations, tutorial system, personalise course, optional modules, resources and teaching quality.
        ''']

diagnosis_2020_positive.loc[2, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Content', '21.9%',
        '''
       learning and support resources, breadth of knowledge and module variety, intellectually stimulating and interesting, career readingess.
        ''']

diagnosis_2020_negative = pd.DataFrame(columns=['Theme', 'Percentage of LDA Tokens', 'Complaint'])
diagnosis_2020_negative.loc[0, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Course', '18.8%',
    '''
    insufficient resources for degree, lack of available year 3 WBS modules, lack of sense of community and support to learn, especially in first year.
    ''']
diagnosis_2020_negative.loc[1, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Teaching & Learning Set-Up', '16.2%',
    '''
  learning set-up doesn't care about mental health, poor resources, lecture capture not accessible for exam revision, teaching strikes, teaching and course delivery (ineligible chalkboard, inability to take notes when lecturers skip through slides quickly), poor communication with other departments and who to go to for help.    
  ''']
diagnosis_2020_negative.loc[2, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Content', '14.6%',
    '''
  adapting to first year content, lack of foresight for year 2 and 3 modules and beyond (Career), mental health support.
    ''']
diagnosis_2020_negative.loc[3, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Teaching', '14.1%',
    '''
    not considerate, no understanding of content structure and applicability, read-off powerpoint slides, unclear marking criterion and teaching objectives, level of preparation and clarity and enthusiasm of lecturers,
    ''']
diagnosis_2020_negative.loc[4, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Course & Content', '13.5%',
    '''
    lack of opportunities to learn technical skills such as Python, VBA, lack of link for modules between 4 different departments, same value different content different degrees, cs group projects "as we didn't have the background of computer science students but our statistics background wasn't much use in the situation"
    ''']
diagnosis_2020_negative.loc[5, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Department', '12.6%',
    '''
    communciation, lack of career guidance early on in degree, lack of standardised resources and where to go for guidance, lack of communication between departments not invited to CS events etc...
    ''']
diagnosis_2020_negative.loc[6, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Teaching', '10.1%',
    '''
    communication, teaching standards, strikes, no consideration to negative feedback. 
    ''']

# Diagnosis 2021 Table
diagnosis_2021_positive = pd.DataFrame(columns=['Theme', 'Percentage of LDA Tokens', 'Strength'])

diagnosis_2021_positive.loc[0, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Course & Content', '38.4%',
        '''
        Challenging course, career opportunities and employability (x2), real-life applications (x2), stimulating and interesting (x2), organised and self-contained notes and modules (x3), seminar support, applied courses, inter-departmental opportunities, access to broad  range of sibjects (x2)
            # beautiful campus + study spaces??, feedback sessions??, office hours?? support office (x2)??, lecturers?? online exams??
    ''']

diagnosis_2021_positive.loc[1, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Community & Department', '35.5%',
        '''
        community spirit (x2), department specialisation in higher years (x2), informative and well-structured communication from department (x6), student community learn from others, campus and stats building (x3), unique courses
            # intellectually stimulating (x2)??, career prospects and qualifications (x3)??
    ''']

diagnosis_2021_positive.loc[2, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Teaching & Learning Set-Up', '26.1%',
        '''
        Teaching & Learning Set-Up: talented staff and helpful (x5), courses bring together concepts, 
            # interesting and challenging course (x4), coursemates and networking (x2)
    ''']


diagnosis_2021_negative = pd.DataFrame(columns=['Theme', 'Percentage of LDA Tokens', 'Complaint'])

diagnosis_2021_negative.loc[0, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Community', '37.4%',
    '''
    lack of care from uni, lack of community, no stats only study space, feel estraneged from CS Department, lack of student-teacher interaction in pre-recorded lectures (x4), first year adaptation, irresponsive to feedback (x4), lack of group work,
        # difficult course? (x2)??
    ''']

diagnosis_2021_negative.loc[1, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Teaching & Learning Set-Up', '25%',
    '''
        course unorganised, online exam preparation, accessibility to lecturers, online teaching (x6), online learning 
        # lack mental health support (x3), career readiness, too many core modules first and second year,
    ''']

diagnosis_2021_negative.loc[2, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Course & Content', '21.1%',
    '''
        module term timetable disbalance (x2), course does not enhance career readiness and employability (x1), struggle to catch up esp if you go behind (x3),
        # library shut + no SU support (x2), online teaching: reading slides + no support and resources (x4)
    ''']

diagnosis_2021_negative.loc[3, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Department', '16.6%',
    '''
        lack of study spaces (x2), lack of personal tutor support, department style of exams care about distributions more than student wellbeing, support office response
        # lack of group work, pre-recorded lectures, expensive parking and bus fares.
    ''']

# Diagnosis 2022 Table
diagnosis_2022_positive = pd.DataFrame(columns=['Theme', 'Percentage of LDA Tokens', 'Strength'])

diagnosis_2022_positive.loc[0, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Community & Department', '50.1%',
        '''• Support staff in the Statistics Department,
        • Workspaces in the department and the Library, 
        • Coursemates come from a very diversified background and easy to become friends with,
        • Self-certification ability and response to the pandemic, 
        • Personal tutor.
        module choices?? good teachers??
    ''']

diagnosis_2022_positive.loc[1, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Teaching & Learning Set-Up', '34.9%',
    '''
    • Lecturers are helpful, supportive, proactive, and responsive to feedback
    • Ample and sufficient resources are provided to aid learning
    • These resources accompany the impressive number of module options to enhance learning
    • Great infrastructure & Facilities
    • Abundant support available to help first years transition to university
    ''']

diagnosis_2022_positive.loc[2, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Course & Content', '15%',
        '''
    • Content of modules are challenging, rewarding, stimulating, and enjoyable
    • Breadth of course is impressive
    • Course enables the student to develop many skills such as problem-solving.
    • Course enables students to find and pursue their interests.
    • Course enables students to learn from departments such as Computer Science, developing relevant programming skills.
    ''']

diagnosis_2022_negative = pd.DataFrame(columns=['Theme', 'Percentage of LDA Tokens', 'Complaint'])

diagnosis_2022_negative.loc[0, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Assessment & Feedback', '28.8%',
    '''• unnecessary core modules (x2), delayed marking, unhelpful feedback (x2), January exam overlap with term 2 (x4), late timetable release for april/summer exams, assessed on memorising facts
    ''']

diagnosis_2022_negative.loc[1, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    '(Online) Teaching & Exams', '28%',
    '''• early response delayed, reluctance to go back to normal (x2), more difficult exams (x5), low online teaching effort, online cheating, not organised modules)
    ''']

diagnosis_2022_negative.loc[2, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Community & Department', '26.2%',
    '''•  lack of teamwork assignment, repeated info in emails, no place to relax at uni, lack of access to study spaces after certain times of the day, poor SU outlets,no community in halls, late release of exam timetable (x2), mental health (x3)
    ''']

diagnosis_2022_negative.loc[3, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Teaching & Learning Set-Up', '17%',
    '''• no sense of community in cohort, no return to f-t-f teaching/online learning: short videos lack of depth (x4), no group work in year 1, no peer assessment so some people did not work (st 340), staff delayed reponse to queries, lack of communication between departments (x2)
    '''
    ]


# Diagnosis 2023 Table
diagnosis_2023_positive = pd.DataFrame(columns=['Theme', 'Percentage of LDA Tokens', 'Strength'])

diagnosis_2023_positive.loc[0, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Community & Department', '36.8%',
        '''• Supportive members of teaching and support staff including teachers (x7),
    • Sudying Facilities (Department and Elsewhere) (x4),
    • Interdisciplinary nature of Department (x4),
    • Student societies (x3),
    • Student environment (x2),
    • Career Support.
    ''']

diagnosis_2023_positive.loc[1, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Course & Content', '33.2%',
    '''• Academic environment,
    • Final Year ability to explore streams such as the Business School,
    • Breadth and Variety of modules (x5),
    • Depth of content, fulfilling and engaging (x4),
    • Cultivate interpersonal skills, grow, and learn,
    • Useful and practical content, enhance career readingess (x2),
    • Courses link together.
    well taught (x2) ??
    ''']

diagnosis_2023_positive.loc[2, ('Theme', 'Percentage of LDA Tokens', 'Strength')] = [
    'Teaching & Learning Set-Up', '30.0%',
    '''• Range of expertise,
    • Mental Health support to aid learning experience (x2),
    • Approachable, helpful & supportive lecturers (x2),
    • Proactive responses to feedback 
    • Hybrid learning set-up during Covid and self-certification facilities (x2),
    • 1st-year tutorials help school to universaty adaption,
    • Face-to-face Learning,
    • Support Office,
    • Personal Tutor Support (x2).
    future opportunities?? course flexibility?? intelectually stimulating (x2)??
    ''']

diagnosis_2023_negative = pd.DataFrame(columns=['Theme', 'Percentage of LDA Tokens', 'Complaint'])

diagnosis_2023_negative.loc[0, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Teaching & Learning Set-Up', '31.2%',
    '''• Infrequent seminars (not every alternate week), 
    • Lack of support from WBS Support Office, • Learning coinciding with January Exams (x2),
    • Lack of support for online teaching and well=being during the pandemic (x4), 
    • Lack of career related support, 
    • Fatigue without a reading week,
    not understandable, feedback (x1)?? restricted learning variety (x2, 1 in year 2)??
    ''']

diagnosis_2023_negative.loc[1, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Course & Content', '25.3%',
    '''• Difficult and challenging to understand (x4), • Harsh grading and difficult assignments (x3),
    • Workload leads to health issues (x3).
    ''']

diagnosis_2023_negative.loc[2, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Learning Inconveniences', '15.6%',
    '''• Late publication of April/Summer Exam timetable,
    • Lack of lecture recordings, especially with inter-departmental module clashes (x2), 
    • Feedback (x2) [Math Department does no provide solution to assignments and exams, 
    Marking is not objective nor clear x3].
    ''']

diagnosis_2023_negative.loc[3, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Department', '14%',
    '''
    • Lack of self-study resources such as lecture recordings,
    • Lack of studying spaces,
    • The department is not welcoming (x2, 1 as disabled), 
    • Late publication of April/Summer Exam timetable,
    • Monitoring points are a waste of time,
    • Department courses have very theoretical and difficult content (x3).
    ''']

diagnosis_2023_negative.loc[4, ('Theme', 'Percentage of LDA Tokens', 'Complaint')] = [
    'Community', '13.9%',
    '''
    • Late publication of April/Summer Exam timetable, 
    • Campus accommodation, gym facilities, and travelling from off-campus are all very expensive,
    • The study environment is very competitive,
    • Poor quality and choice of campus cafes & restaurants.
    career readiness connections??
    ''']


page1 = """
# Student Learning Experience
###### NSS Data 2015-2023, Statistics Department

### Themes- What are the students talking about?
###### Positive Themes
<|layout|columns= 1 1 1|
    <|
Community & Department
<|{df_community_department_positive}|chart|type=bar|x=Year|y[1]=Community|y[2]=Department|y[3]=Community & Department|type[3]=line|layout={layout_longitudinal}|>

    |>

    <|
Course & Content
<|{df_course_content_positive}|chart|type=bar|x=Year|y[1]=Course & Content|type[1]=line|layout={layout_longitudinal}|>
    |>

    <|
Teaching & Learning Set-Up
<|{df_teaching_learning_setup_positive}|chart|type=bar|x=Year|y[1]=Teaching & Learning Set-Up|type[1]=line|layout={layout_longitudinal}|>
    |>
|>


###### Negative Themes
<|layout|columns= 1 1 1|
    <|
Community & Department
<|{df_community_department_negative}|chart|type=bar|x=Year|y[1]=Community|y[2]=Department|y[3]=Community & Department|type[3]=line|color[3]=red|layout={layout_longitudinal}|>
    |>

    <|
Course & Content
<|{df_course_content_negative}|chart|type=bar|x=Year|y[1]=Course & Content|type[1]=line|color[1]=red|layout={layout_longitudinal}|>
    |>

    <|
Teaching & Learning Set-Up
<|{df_teaching_learning_setup_negative}|chart|type=bar|x=Year|y[1]=Assessment & Feedback|y[2]=Teaching & Exams|y[3]=Learning Set-Up|y[4]=Teaching & Learning Set-Up|type[4]=line|color[4]=red|layout={layout_longitudinal}|>
    |>
|>


### Diagnosis (Strengths & Areas for Improvement)
<|2014/15|expandable|expanded=False|
<|layout|columns= 1 1|
    <|
Positive
<|{diagnosis_2015_positive}|table|>
    |>

    <|
Negative
<|{diagnosis_2015_negative}|table|>
    |>
|>
|>

<|2015/16|expandable|expanded=False|
<|layout|columns= 1 1|
    <|
Positive
<|{diagnosis_2016_positive}|table|>
    |>

    <|
Negative
<|{diagnosis_2016_negative}|table|>
    |>
|>
|>

<|2016/17|expandable|expanded=False|
<|layout|columns= 1 1|
    <|
Positive
<|{diagnosis_2017_positive}|table|>
    |>

    <|
Negative
<|{diagnosis_2017_negative}|table|>
    |>
|>
|>

<|2017/18|expandable|expanded=False|
<|layout|columns= 1 1|
    <|
Positive
<|{diagnosis_2018_positive}|table|>
    |>

    <|
Negative
<|{diagnosis_2018_negative}|table|>
    |>
|>
|>

<|2018/19|expandable|expanded=False|
<|layout|columns= 1 1|
    <|
Positive
<|{diagnosis_2019_positive}|table|>
    |>

    <|
Negative
<|{diagnosis_2019_negative}|table|>
    |>
|>
|>

<|2019/20|expandable|expanded=False|
<|layout|columns= 1 1|
    <|
Positive
<|{diagnosis_2020_positive}|table|>
    |>

    <|
Negative
<|{diagnosis_2020_negative}|table|>
    |>
|>
|>


<|2020/21|expandable|expanded=False|
<|layout|columns= 1 1|
    <|
Positive
<|{diagnosis_2021_positive}|table|>
    |>

    <|
Negative
<|{diagnosis_2021_negative}|table|>
    |>
|>
|>

<|2021/22|expandable|expanded=False|
<|layout|columns= 1 1|
    <|
Positive
<|{diagnosis_2022_positive}|table|>
    |>

    <|
Negative
<|{diagnosis_2022_negative}|table|>
    |>
|>
|>

<|2022/23|expandable|expanded=False|
<|layout|columns= 1 1|
    <|
Positive
<|{diagnosis_2023_positive[['Theme', 'Percentage of LDA Tokens', 'Strength']]}|table|>
    |>

    <|
Negative
<|{diagnosis_2023_negative[['Theme', 'Percentage of LDA Tokens', 'Complaint']]}|table|>
    |>
|>
|>


### Trends
<|Emerging Themes|expandable|expanded=False|
<|{diagnosis_2015_positive}|table|>
|>

<|Recurrent Themes|expandable|expanded=False|
<|{diagnosis_2015_positive}|table|>
|>

<|Cyclical Themes|expandable|expanded=False|
<|{diagnosis_2015_positive}|table|>
|>

"""


# Function to split comments into sentences and create new rows
def expand_comments(df, column_keep, column_name):
    sentences = []
    indices = []
    for index, row in df.iterrows():
        course = row[column_keep]
        comment = row[column_name]
        for sentence in nltk.sent_tokenize(comment):
            sentences.append(sentence)
            indices.append(index)
    try:
        return pd.DataFrame({'Department': df.loc[indices, column_keep[0]], 
                         'Course': df.loc[indices, column_keep[1]],
                         column_name: sentences})
    except:
        return pd.DataFrame({'Department': df.loc[indices, column_keep[0]], 
                         column_name: sentences}) 

# 2014/15 Page
df_15 = pd.read_excel("input/NSS - National Student Survey/2015/ST student comments.xlsx", 
                        header=[14], sheet_name=None)
df_2015 = df_15['Comments'].copy()
df_2015.rename(columns={"Positive comment": "Positive", "Negative comment": 'Negative',
                        "Institution own comment": "Improve Experience"}, inplace=True)
df_2015['non_tokenised_negative'] = df_2015['Negative'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
df_2015['non_tokenised_positive'] = df_2015['Positive'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
empty = {'Negative': '', 'Positive': '', 
         "Improve Experience": ''}
df_2015.fillna(value=empty, inplace=True)
df_groupedreviews2015 = df_2015.groupby('Department 1')[['Negative', 'Positive', "Improve Experience"]].agg(' '.join)

# Preview Dataset Input
df_2015_present = df_2015[['Department 1', 'Negative', 'Positive']]

positive_df_2015 = expand_comments(df_2015[['Department 1', 'Negative', 'Positive']], 
                              ['Department 1'], 'Positive')
negative_df_2015 = expand_comments(df_2015[['Department 1', 'Negative', 'Positive']], 
                              ['Department 1'], 'Negative')
positive_df_2015 = positive_df_2015[~positive_df_2015.Positive.str.contains('No Response Entered')]
negative_df_2015 = negative_df_2015[~negative_df_2015.Negative.str.contains('No Response Entered')]
# Dataset example
row_info_2015 = f"This dataset has {df_2015.shape[0]} entries."

# Input by list
positive_list_15 = positive_df_2015['Positive'].values.tolist()
negative_list_15 = negative_df_2015['Negative'].values.tolist()
list_by_sentiment_2015 = [positive_list_15, negative_list_15]

# Input by string
positive_text_15 = ' '.join(positive_df_2015['Positive'])
negative_text_15 = ' '.join(negative_df_2015['Negative'])
text_by_sentiment_2015 = [positive_text_15, negative_text_15]

# PyLDA Model
positive_lda_topics15 = get_lda_topics(positive_list_15, 2)
negative_lda_topics15 = get_lda_topics(negative_list_15, 3)

#NMF Model
positive_nmf_topics15 = get_nmf_topics(positive_list_15, 2)
negative_nmf_topics15 = get_nmf_topics(negative_list_15, 3)

#Luhn and LexRank Extractive Summary Models
positive_extractive_summaries_15 = get_extractive_summaries(positive_text_15)
negative_extractive_summaries_15 = get_extractive_summaries(negative_text_15)

# Get the dominant topic and its probability for each comment
dominant_topics, dominant_probabilities = get_dominant_topic(positive_list_15, 2)

# Add the dominant topic column to the dataset
positive_df_2015['Positive Dominant Topic'] = dominant_topics
positive_df_2015['Positive Dominant Topic Probability'] = dominant_probabilities
positive_df_2015['Positive Dominant Topic Probability'] = round(positive_df_2015['Positive Dominant Topic Probability'], 2)
positive_df_2015['Positive Dominant Topic Probability'] = positive_df_2015['Positive Dominant Topic Probability'].astype(str)

# Get the dominant topic and its probability for each comment
dominant_topics, dominant_probabilities = get_dominant_topic(negative_list_15, 3)

# Add the dominant topic column to the dataset
negative_df_2015['Negative Dominant Topic'] = dominant_topics
negative_df_2015['Negative Dominant Topic Probability'] = dominant_probabilities
negative_df_2015['Negative Dominant Topic Probability'] = round(negative_df_2015['Negative Dominant Topic Probability'], 2)
negative_df_2015['Negative Dominant Topic Probability'] = negative_df_2015['Negative Dominant Topic Probability'].astype(str)

themes_2015 = [
        "Content & Modules", "Teaching"
]

themes_2015_negative = [
        "Content", "Teaching & Learning Set-Up"
]

themes_data_2015 = [
    {
        # Values for Positive Themes
        "values": [55.1, 44.9],
        "labels": themes_2015
    },
    {
        # Values for Negative Themes
        "values": [37.3, 62.7],
        "labels": themes_2015_negative
    }
]


page2 = """
# Evaluating Learning For the 2014/15 Academic Year
#### Themes Overview
<|{themes_data_2015}|chart|type=pie|x[1]=0/values|x[2]=1/values|options={options_piechart_thematic}|layout={layout_piechart_thematic}|>

#### Strengths

##### Strength 1: Content & Modules (55.1%)
<|{diagnosis_2015_positive.loc[0:0]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2015[positive_df_2015['Positive Dominant Topic'].isin([1])])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2015[positive_df_2015['Positive Dominant Topic'].isin([1])]}|table|> 
|>


##### Strength 2: Teaching (44.9%)
<|{diagnosis_2015_positive.loc[1:1]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2015[positive_df_2015['Positive Dominant Topic'].isin([2])])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2015[positive_df_2015['Positive Dominant Topic'].isin([2])]}|table|> 
|>


#### Areas for Improvement (AFI)

##### AFI 1: Content (37.3%)
<|{diagnosis_2015_negative.loc[0:0]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2015[negative_df_2015['Negative Dominant Topic'].isin([2])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2015[negative_df_2015['Negative Dominant Topic'].isin([2])]}|table|> 
|>


##### AFI 2: Teaching & Learning Set-Up (62.7%)
<|{diagnosis_2015_negative.loc[0:0]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2015[negative_df_2015['Negative Dominant Topic'].isin([1,3])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2015[negative_df_2015['Negative Dominant Topic'].isin([1,3])]}|table|> 
|>

"""

# 2015/16 Page
df_16 = pd.read_excel("input/NSS - National Student Survey/2016/All comments (ST).xlsx", 
                        header=[0], sheet_name=None)
# df_16.keys()
df_2016 = df_16['Comments'].copy()
df_2016.rename(columns={"Positive comment": "Positive", "Negative comment": 'Negative', "Institution own comment: 'What is the one thing we could have done to improve your overall experience?'": "Improve Experience"}, inplace=True)
df_2016['non_tokenised_negative'] = df_2016['Negative'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
df_2016['non_tokenised_positive'] = df_2016['Positive'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
empty = {'Negative': '', 'Positive': '', 
         "Improve Experience": ''}
df_2016.fillna(value=empty, inplace=True)
df_groupedreviews2016 = df_2016.groupby('Course code')[['Negative', 'Positive']].agg(' '.join)

positive_df_2016 = expand_comments(df_2016[['Department', 'Course code', 'Negative', 'Positive']], 
                              ['Department', 'Course code'], 'Positive')
negative_df_2016 = expand_comments(df_2016[['Department','Course code', 'Negative', 'Positive']], 
                              ['Department', 'Course code'], 'Negative')
positive_df_2016 = positive_df_2016[~positive_df_2016.Positive.str.contains('No Response Entered')]
negative_df_2016 = negative_df_2016[~negative_df_2016.Negative.str.contains('No Response Entered')]

# Input by list
positive_list_16 = positive_df_2016['Positive'].values.tolist()
negative_list_16 = negative_df_2016['Negative'].values.tolist()
list_by_sentiment_2016 = [positive_list_16, negative_list_16]

# Input by string
positive_text_16 = ' '.join(positive_df_2016['Positive'])
negative_text_16 = ' '.join(negative_df_2016['Negative'])
text_by_sentiment_2016 = [positive_text_16, negative_text_16]

# PyLDA Model
positive_lda_topics16 = get_lda_topics(positive_list_16, 7)
negative_lda_topics16 = get_lda_topics(negative_list_16, 6)

#NMF Model
positive_nmf_topics16 = get_nmf_topics(positive_list_16, 7)
negative_nmf_topics16 = get_nmf_topics(negative_list_16, 6)

#Luhn and LexRank Extractive Summary Models
positive_extractive_summaries_16 = get_extractive_summaries(positive_text_16)
negative_extractive_summaries_16 = get_extractive_summaries(negative_text_16)

# Get the dominant topic and its probability for each comment
dominant_topics, dominant_probabilities = get_dominant_topic(positive_list_16, 7)

# Add the dominant topic column to the dataset
positive_df_2016['Positive Dominant Topic'] = dominant_topics
positive_df_2016['Positive Dominant Topic Probability'] = dominant_probabilities
positive_df_2016['Positive Dominant Topic Probability'] = round(positive_df_2016['Positive Dominant Topic Probability'], 2)
positive_df_2016['Positive Dominant Topic Probability'] = positive_df_2016['Positive Dominant Topic Probability'].astype(str)

# Dataset example
row_info_2016 = f"This dataset has {positive_df_2016.shape[0]} entries."

# Get the dominant topic and its probability for each comment
dominant_topics, dominant_probabilities = get_dominant_topic(negative_list_16, 6)

# Add the dominant topic column to the dataset
negative_df_2016['Negative Dominant Topic'] = dominant_topics
negative_df_2016['Negative Dominant Topic Probability'] = dominant_probabilities
negative_df_2016['Negative Dominant Topic Probability'] = round(negative_df_2016['Negative Dominant Topic Probability'], 2)
negative_df_2016['Negative Dominant Topic Probability'] = negative_df_2016['Negative Dominant Topic Probability'].astype(str)

themes_2016 = [
        "Modules", "Content", "Course", "Department"
]

themes_2016_negative = [
        "First Year Adaptation & Guidance", "Teaching", "Course", "Community", "Content"
]

themes_data_2016 = [
    {
        # Values for Positive Themes
        "values": [35.2, 18.2, 39, 7.6],
        "labels": themes_2016
    },
    {
        # Values for Negative Themes
        "values": [38.3, 21, 17.8, 12.5, 10.4],
        "labels": themes_2016_negative
    }
]


page3 = """
# Evaluating Learning For the 2015/16 Academic Year
#### Themes Overview
<|{themes_data_2016}|chart|type=pie|x[1]=0/values|x[2]=1/values|options={options_piechart_thematic}|layout={layout_piechart_thematic}|>

#### Strengths

##### Strength 1: Modules (35.2%)
<|{diagnosis_2016_positive.loc[:0]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2016[positive_df_2016['Positive Dominant Topic'].isin([1,7])])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2016[positive_df_2016['Positive Dominant Topic'].isin([1,7])]}|table|> 
|>

##### Strength 2: Content (18.2%)
<|{diagnosis_2016_positive.loc[1:1]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2016[positive_df_2016['Positive Dominant Topic'].isin([4])])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2016[positive_df_2016['Positive Dominant Topic'].isin([4])]}|table|> 
|>

##### Strength 3: Course (39%)
<|{diagnosis_2016_positive.loc[2:2]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2016[positive_df_2016['Positive Dominant Topic'].isin([6,3,2])])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2016[positive_df_2016['Positive Dominant Topic'].isin([6,3,2])]}|table|> 
|>

##### Strength 4: Department (7.6%)
<|{diagnosis_2016_positive.loc[3:3]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2016[positive_df_2016['Positive Dominant Topic'].isin([5])])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2016[positive_df_2016['Positive Dominant Topic'].isin([5])]}|table|> 
|>

#### Areas for Improvement (AFI)

##### AFI 1: First Year Adaptation & Guidance (38.3%)
<|{diagnosis_2016_negative.loc[0:0]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2016[negative_df_2016['Negative Dominant Topic'].isin([5,3])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2016[negative_df_2016['Negative Dominant Topic'].isin([5,3])]}|table|> 
|>

##### AFI 2: Teaching (21%)
<|{diagnosis_2016_negative.loc[1:1]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2016[negative_df_2016['Negative Dominant Topic'].isin([2])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2016[negative_df_2016['Negative Dominant Topic'].isin([2])]}|table|> 
|>

##### AFI 3: Course (17.8%)
<|{diagnosis_2016_negative.loc[2:2]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2016[negative_df_2016['Negative Dominant Topic'].isin([4])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2016[negative_df_2016['Negative Dominant Topic'].isin([4])]}|table|> 
|>

##### AFI 4: Community (12.5%)
<|{diagnosis_2016_negative.loc[3:3]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2016[negative_df_2016['Negative Dominant Topic'].isin([1])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2016[negative_df_2016['Negative Dominant Topic'].isin([1])]}|table|> 
|>

##### AFI 5: Content (10.4%)
<|{diagnosis_2016_negative.loc[4:4]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2016[negative_df_2016['Negative Dominant Topic'].isin([6])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2016[negative_df_2016['Negative Dominant Topic'].isin([6])]}|table|> 
|>

"""


# 2016/17 Page
df_17 = pd.read_excel("input/NSS - National Student Survey/2017/STATISTICS comments and Qs.xlsx", 
                        header=[0], sheet_name=None)
# df_17.keys()
df_2017 = df_17['Comments (2)'].copy()
df_2017.rename(columns={"Positive comment": "Positive", "Negative comment": 'Negative',                       "Institution own comment: 'What is the one thing we could have done to improve your overall experience?'": "Improve Experience"}, inplace=True)
df_2017['non_tokenised_negative'] = df_2017['Negative'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
df_2017['non_tokenised_positive'] = df_2017['Positive'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
empty = {'Negative': '', 'Positive': '', 
         "Improve Experience": ''}
df_2017.fillna(value=empty, inplace=True)
df_groupedreviews2017 = df_2017.groupby('course:')[['Negative', 'Positive']].agg(' '.join)

positive_df_2017 = expand_comments(df_2017[['Department', 'course:', 'Negative', 'Positive']], 
                              ['Department', 'course:'], 'Positive')
negative_df_2017 = expand_comments(df_2017[['Department','course:', 'Negative', 'Positive']], 
                              ['Department', 'course:'], 'Negative')
positive_df_2017 = positive_df_2017[~positive_df_2017.Positive.str.contains('No Response Entered')]
negative_df_2017 = negative_df_2017[~negative_df_2017.Negative.str.contains('No Response Entered')]

# Input by list
positive_list_17 = positive_df_2017['Positive'].values.tolist()
negative_list_17 = negative_df_2017['Negative'].values.tolist()
list_by_sentiment_2017 = [positive_list_17, negative_list_17]

# Input by string
positive_text_17 = ' '.join(positive_df_2017['Positive'])
negative_text_17 = ' '.join(negative_df_2017['Negative'])
text_by_sentiment_2017 = [positive_text_17, negative_text_17]

# PyLDA Model
positive_lda_topics17 = get_lda_topics(positive_list_17, 3)
negative_lda_topics17 = get_lda_topics(negative_list_17, 3)

#NMF Model
positive_nmf_topics17 = get_nmf_topics(positive_list_17, 3)
negative_nmf_topics17 = get_nmf_topics(negative_list_17, 3)

#Luhn and LexRank Extractive Summary Models
positive_extractive_summaries_17 = get_extractive_summaries(positive_text_17)
negative_extractive_summaries_17 = get_extractive_summaries(negative_text_17)

# Get the dominant topic and its probability for each comment
dominant_topics, dominant_probabilities = get_dominant_topic(positive_list_17, 3)

# Add the dominant topic column to the dataset
positive_df_2017['Positive Dominant Topic'] = dominant_topics
positive_df_2017['Positive Dominant Topic Probability'] = dominant_probabilities
positive_df_2017['Positive Dominant Topic Probability'] = round(positive_df_2017['Positive Dominant Topic Probability'], 2)
positive_df_2017['Positive Dominant Topic Probability'] = positive_df_2017['Positive Dominant Topic Probability'].astype(str)

# Dataset example
row_info_2017 = f"This dataset has {positive_df_2017.shape[0]} entries."

# Get the dominant topic and its probability for each comment
dominant_topics, dominant_probabilities = get_dominant_topic(negative_list_17, 3)

# Add the dominant topic column to the dataset
negative_df_2017['Negative Dominant Topic'] = dominant_topics
negative_df_2017['Negative Dominant Topic Probability'] = dominant_probabilities
negative_df_2017['Negative Dominant Topic Probability'] = round(negative_df_2017['Negative Dominant Topic Probability'], 2)
negative_df_2017['Negative Dominant Topic Probability'] = negative_df_2017['Negative Dominant Topic Probability'].astype(str)

themes_2017 = [
        "Course", "Modules", "Learning Set-Up"
]

themes_2017_negative = [
        "Course", "Modules", "Learning Set-Up"
]

themes_data_2017 = [
    {
        # Values for Positive Themes
        "values": [45.4, 37.7, 16.9],
        "labels": themes_2017
    },
    {
        # Values for Negative Themes
        "values": [43.6, 29.2, 27.3],
        "labels": themes_2017_negative
    }
]


page4 = """
# Evaluating Learning For the 2016/17 Academic Year
#### Themes Overview
<|{themes_data_2017}|chart|type=pie|x[1]=0/values|x[2]=1/values|options={options_piechart_thematic}|layout={layout_piechart_thematic}|>

#### Strengths

##### Strength 1: Course (45.4%)
<|{diagnosis_2017_positive.loc[:0]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2017[positive_df_2017['Positive Dominant Topic'].isin([1])])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2017[positive_df_2017['Positive Dominant Topic'].isin([1])]}|table|> 
|>

##### Strength 2: Modules (37.7%)
<|{diagnosis_2017_positive.loc[1:1]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2017[positive_df_2017['Positive Dominant Topic'].isin([2])])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2017[positive_df_2017['Positive Dominant Topic'].isin([2])]}|table|> 
|>

##### Strength 3: Learning Set-Up (16.9%)
<|{diagnosis_2017_positive.loc[2:2]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2017[positive_df_2017['Positive Dominant Topic'].isin([3])])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2017[positive_df_2017['Positive Dominant Topic'].isin([3])]}|table|> 
|>

#### Areas for Improvement (AFI)

##### AFI 1: Course (43.6%)
<|{diagnosis_2017_negative.loc[0:0]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2017[negative_df_2017['Negative Dominant Topic'].isin([3])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2017[negative_df_2017['Negative Dominant Topic'].isin([3])]}|table|> 
|>

##### AFI 2: Modules (29.2%)
<|{diagnosis_2017_negative.loc[1:1]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2017[negative_df_2017['Negative Dominant Topic'].isin([1])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2017[negative_df_2017['Negative Dominant Topic'].isin([1])]}|table|> 
|>

##### AFI 3: Learning Set-Up (27.3%)
<|{diagnosis_2017_negative.loc[2:2]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2017[negative_df_2017['Negative Dominant Topic'].isin([2])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2017[negative_df_2017['Negative Dominant Topic'].isin([2])]}|table|> 
|>
"""


# 2017/18 Page
df_18 = pd.read_excel("input/NSS - National Student Survey/2018/NSS Comments 2018 - Untagged - ST - Statistics - v2.xlsx", 
                        header=[0], sheet_name=None)

# df_18.keys()
df_2018 = df_18['Sheet1'].copy()
df_2018.rename(columns={"Positive comment": "Positive", "Negative comment": 'Negative', "Institution own comment: 'What is the one thing we could have done to improve your overall experience?'": "Improve Experience"}, inplace=True)
df_2018['non_tokenised_negative'] = df_2018['Negative'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
df_2018['non_tokenised_positive'] = df_2018['Positive'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
empty = {'Negative': '', 'Positive': '', 
         "Improve Experience": ''}
df_2018.fillna(value=empty, inplace=True)
df_groupedreviews2018 = df_2018.groupby('Course title')[['Negative', 'Positive']].agg(' '.join)

positive_df_2018 = expand_comments(df_2018[['Dept', 'Course title', 'Negative', 'Positive']], 
                              ['Dept', 'Course title'], 'Positive')
negative_df_2018 = expand_comments(df_2018[['Dept','Course title', 'Negative', 'Positive']], 
                              ['Dept', 'Course title'], 'Negative')
positive_df_2018 = positive_df_2018[~positive_df_2018.Positive.str.contains('No Response Entered')]
negative_df_2018 = negative_df_2018[~negative_df_2018.Negative.str.contains('No Response Entered')]

# Input by list
positive_list_18 = positive_df_2018['Positive'].values.tolist()
negative_list_18 = negative_df_2018['Negative'].values.tolist()
list_by_sentiment_2018 = [positive_list_18, negative_list_18]

# Input by string
positive_text_18 = ' '.join(positive_df_2018['Positive'])
negative_text_18 = ' '.join(negative_df_2018['Negative'])
text_by_sentiment_2018 = [positive_text_18, negative_text_18]

# PyLDA Model
positive_lda_topics18 = get_lda_topics(positive_list_18, 3)
negative_lda_topics18 = get_lda_topics(negative_list_18, 6)

#NMF Model
positive_nmf_topics18 = get_nmf_topics(positive_list_18, 3)
negative_nmf_topics18 = get_nmf_topics(negative_list_18, 6)

#Luhn and LexRank Extractive Summary Models
positive_extractive_summaries_18 = get_extractive_summaries(positive_text_18)
negative_extractive_summaries_18 = get_extractive_summaries(negative_text_18)

# Get the dominant topic and its probability for each comment
dominant_topics, dominant_probabilities = get_dominant_topic(positive_list_18, 3)

# Add the dominant topic column to the dataset
positive_df_2018['Positive Dominant Topic'] = dominant_topics
positive_df_2018['Positive Dominant Topic Probability'] = dominant_probabilities
positive_df_2018['Positive Dominant Topic Probability'] = round(positive_df_2018['Positive Dominant Topic Probability'], 2)
positive_df_2018['Positive Dominant Topic Probability'] = positive_df_2018['Positive Dominant Topic Probability'].astype(str)

# Dataset example
row_info_2018 = f"This dataset has {positive_df_2018.shape[0]} entries."

# Get the dominant topic and its probability for each comment
dominant_topics, dominant_probabilities = get_dominant_topic(negative_list_18, 6)

# Add the dominant topic column to the dataset
negative_df_2018['Negative Dominant Topic'] = dominant_topics
negative_df_2018['Negative Dominant Topic Probability'] = dominant_probabilities
negative_df_2018['Negative Dominant Topic Probability'] = round(negative_df_2018['Negative Dominant Topic Probability'], 2)
negative_df_2018['Negative Dominant Topic Probability'] = negative_df_2018['Negative Dominant Topic Probability'].astype(str)

themes_2018 = [
        "Course", "Teaching", "Learning Set-Up"
]

themes_2018_negative = [
        "Department", "Course", "Learning Set-Up", "Teaching"
]

themes_data_2018 = [
    {
        # Values for Positive Themes
        "values": [36.2, 33.6, 30.2],
        "labels": themes_2018
    },
    {
        # Values for Negative Themes
        "values": [24.8,17.9,43.7,13.7],
        "labels": themes_2018_negative
    }
]


page5 = """
# Evaluating Learning For the 2017/18 Academic Year
#### Themes Overview
<|{themes_data_2018}|chart|type=pie|x[1]=0/values|x[2]=1/values|options={options_piechart_thematic}|layout={layout_piechart_thematic}|>

#### Strengths

##### Strength 1: Course (36.2%)
<|{diagnosis_2018_positive.loc[:0]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2018[positive_df_2018['Positive Dominant Topic'].isin([1])])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2018[positive_df_2018['Positive Dominant Topic'].isin([1])]}|table|> 
|>

##### Strength 2: Teaching (33.6%)
<|{diagnosis_2018_positive.loc[1:1]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2018[positive_df_2018['Positive Dominant Topic'].isin([2])])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2018[positive_df_2018['Positive Dominant Topic'].isin([2])]}|table|> 
|>


##### Strength 3: Learning Set-Up (30.2%)
<|{diagnosis_2018_positive.loc[2:2]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2018[positive_df_2018['Positive Dominant Topic'].isin([3])])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2018[positive_df_2018['Positive Dominant Topic'].isin([3])]}|table|> 
|>


#### Areas for Improvement (AFI)

##### AFI 1: Department (24.8%)
<|{diagnosis_2018_negative.loc[0:0]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2018[negative_df_2018['Negative Dominant Topic'].isin([1])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2018[negative_df_2018['Negative Dominant Topic'].isin([1])]}|table|> 
|>

##### AFI 2: Course (17.9%)
<|{diagnosis_2018_negative.loc[1:1]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2018[negative_df_2018['Negative Dominant Topic'].isin([4])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2018[negative_df_2018['Negative Dominant Topic'].isin([4])]}|table|> 
|>

##### AFI 3: Learning Set-Up (43.7%)
<|{diagnosis_2018_negative.loc[2:2]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2018[negative_df_2018['Negative Dominant Topic'].isin([2,6,5])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2018[negative_df_2018['Negative Dominant Topic'].isin([2,6,5])]}|table|> 
|>

##### AFI 4: Teaching (13.7%)
<|{diagnosis_2018_negative.loc[3:3]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2018[negative_df_2018['Negative Dominant Topic'].isin([3])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2018[negative_df_2018['Negative Dominant Topic'].isin([3])]}|table|> 
|>



"""


# 2018/19 Page
df_19 = pd.read_excel("input/NSS - National Student Survey/2019/NSS Comments 2019 – All Comments - ST - Statistics.xlsx", 
                        header=[1], sheet_name=None)
# df_19.keys()
df_2019 = df_19['2019 NSS Student Comments'].copy()
df_2019.rename(columns={"Positive comment": "Positive", "Negative comment": 'Negative',
                        "Institution own comment  (Optional questions)": "Improve Experience"}, inplace=True)

df_2019['non_tokenised_negative'] = df_2019['Negative'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
df_2019['non_tokenised_positive'] = df_2019['Positive'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
df_2019['non_tokenised_improve_experience'] = df_2019['Improve Experience'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
empty = {'Negative': '', 'Positive': '', 
         "Improve Experience": ''}
df_2019.fillna(value=empty, inplace=True)
df_groupedreviews2019 = df_2019.groupby('Course title')[['Negative', 'Positive', "Improve Experience"]].agg(' '.join)

positive_df_2019 = expand_comments(df_2019[['Department', 'Course title', 'Negative', 'Positive', 'Improve Experience']], 
                              ['Department', 'Course title'], 'Positive')
negative_df_2019 = expand_comments(df_2019[['Department','Course title', 'Negative', 'Positive', 'Improve Experience']], 
                              ['Department', 'Course title'], 'Negative')
improve_df_2019 = expand_comments(df_2019[['Department','Course title', 'Negative', 'Positive', 'Improve Experience']], 
                              ['Department', 'Course title'], 'Improve Experience')
positive_df_2019 = positive_df_2019[~positive_df_2019.Positive.str.contains('No Response Entered')]
negative_df_2019 = negative_df_2019[~negative_df_2019.Negative.str.contains('No Response Entered')]
improve_df_2019 = improve_df_2019[~improve_df_2019['Improve Experience'].str.contains('No Response Entered')]

# Input by list
positive_list_19 = positive_df_2019['Positive'].values.tolist()
negative_list_19 = negative_df_2019['Negative'].values.tolist()
improve_experience_list_19 = improve_df_2019['Improve Experience'].values.tolist()
list_by_sentiment_2019 = [positive_list_19, negative_list_19, improve_experience_list_19]

# Input by string
positive_text_19 = ' '.join(positive_df_2019['Positive'])
negative_text_19 = ' '.join(negative_df_2019['Negative'])
improve_experience_text_19 = ' '.join(improve_df_2019['Improve Experience'])
text_by_sentiment_2019 = [positive_text_19, negative_text_19, improve_experience_text_19]

# PyLDA Model
positive_lda_topics19 = get_lda_topics(positive_list_19, 7)
negative_lda_topics19 = get_lda_topics(negative_list_19, 8)
improve_lda_topics19 = get_lda_topics(improve_experience_list_19, 2)

#NMF Model
positive_nmf_topics19 = get_nmf_topics(positive_list_19, 7)
negative_nmf_topics19 = get_nmf_topics(negative_list_19, 8)
improve_nmf_topics19 = get_nmf_topics(improve_experience_list_19, 2)

#Luhn and LexRank Extractive Summary Models
positive_extractive_summaries_19 = get_extractive_summaries(positive_text_19)
negative_extractive_summaries_19 = get_extractive_summaries(negative_text_19)
improve_extractive_summaries_19 = get_extractive_summaries(improve_experience_text_19)

# Get the dominant topic and its probability for each comment
dominant_topics, dominant_probabilities = get_dominant_topic(positive_list_19, 7)

# Add the dominant topic column to the dataset
positive_df_2019['Positive Dominant Topic'] = dominant_topics
positive_df_2019['Positive Dominant Topic Probability'] = dominant_probabilities
positive_df_2019['Positive Dominant Topic Probability'] = round(positive_df_2019['Positive Dominant Topic Probability'], 2)
positive_df_2019['Positive Dominant Topic Probability'] = positive_df_2019['Positive Dominant Topic Probability'].astype(str)

# Get the dominant topic and its probability for each comment
dominant_topics, dominant_probabilities = get_dominant_topic(negative_list_19, 8)

# Add the dominant topic column to the dataset
negative_df_2019['Negative Dominant Topic'] = dominant_topics
negative_df_2019['Negative Dominant Topic Probability'] = dominant_probabilities
negative_df_2019['Negative Dominant Topic Probability'] = round(negative_df_2019['Negative Dominant Topic Probability'], 2)
negative_df_2019['Negative Dominant Topic Probability'] = negative_df_2019['Negative Dominant Topic Probability'].astype(str)

# Dataset example
row_info_2019 = f"This dataset has {positive_df_2019.shape[0]} entries."

themes_2019 = [
        "Modules", "Content", "Department", "Community", "Course"
]

themes_2019_negative = [
        "Teaching & Learning Set-Up", "Modules", "Content", "Exam"
]

themes_data_2019 = [
    {
        # Values for Positive Themes
        "values": [32.5,16.4,15.5,15.5,20],
        "labels": themes_2019
    },
    {
        # Values for Negative Themes
        "values": [45.9,23,23.8,7.3],
        "labels": themes_2019_negative
    }
]

page6 =  """
# Evaluating Learning For the 2018/19 Academic Year
#### Themes Overview
<|{themes_data_2019}|chart|type=pie|x[1]=0/values|x[2]=1/values|options={options_piechart_thematic}|layout={layout_piechart_thematic}|>

#### Strengths

##### Strength 1: Modules (32.5%)
<|{diagnosis_2019_positive.loc[:0]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2019[positive_df_2019['Positive Dominant Topic'].isin([1,3])])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2019[positive_df_2019['Positive Dominant Topic'].isin([1,3])]}|table|> 
|>

##### Strength 2: Content (16.4%)
<|{diagnosis_2019_positive.loc[1:1]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2019[positive_df_2019['Positive Dominant Topic']==4])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2019[positive_df_2019['Positive Dominant Topic'].isin([4])]}|table|> 
|>

##### Strength 3: Department (15.5%)
<|{diagnosis_2019_positive.loc[2:2]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2019[positive_df_2019['Positive Dominant Topic']==7])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2019[positive_df_2019['Positive Dominant Topic'].isin([7])]}|table|> 
|>

##### Strength 4: Community (15.5%)
<|{diagnosis_2019_positive.loc[2:2]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2019[positive_df_2019['Positive Dominant Topic']==6])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2019[positive_df_2019['Positive Dominant Topic'].isin([6])]}|table|> 
|>

##### Strength 5: Course (20%)
<|{diagnosis_2019_positive.loc[2:2]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2019[positive_df_2019['Positive Dominant Topic'].isin([2,5])])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2019[positive_df_2019['Positive Dominant Topic'].isin([2,5])]}|table|> 
|>

#### Areas for Improvement (AFI)

##### AFI 1: Teaching & Learning Set-Up (45.9%)
<|{diagnosis_2019_negative.loc[[0,1]]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2019[negative_df_2019['Negative Dominant Topic'].isin([8,3,1])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2019[negative_df_2019['Negative Dominant Topic'].isin([8,3,1])]}|table|> 
|>

##### AFI 2: Modules (23%)
<|{diagnosis_2019_negative.loc[2:2]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2019[negative_df_2019['Negative Dominant Topic'].isin([5,7])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2019[negative_df_2019['Negative Dominant Topic'].isin([5,7])]}|table|> 
|>

##### AFI 3: Content (23.8%)
<|{diagnosis_2019_negative.loc[3:3]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2019[negative_df_2019['Negative Dominant Topic'].isin([4,2])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2019[negative_df_2019['Negative Dominant Topic'].isin([4,2])]}|table|> 
|>

##### AFI 4: Exam (7.3%)
<|{diagnosis_2019_negative.loc[4:4]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2019[negative_df_2019['Negative Dominant Topic'].isin([6])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2019[negative_df_2019['Negative Dominant Topic'].isin([6])]}|table|> 
|>




"""


# 2019/20 Page
df_20 = pd.read_excel("input/NSS - National Student Survey/2020/NSS 2020 Student Comments - Statistics.xlsx", 
                        header=[0], sheet_name=None)
# df_20.keys()
df_2020 = df_20['NSS 2020 Comments'].copy()
df_2020.rename(columns={"Positive comment": "Positive", "Negative comment": 'Negative',                       
                        "Institution own comment": "Improve Experience"}, inplace=True)

df_2020['non_tokenised_negative'] = df_2020['Negative'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
df_2020['non_tokenised_positive'] = df_2020['Positive'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
df_2020['non_tokenised_improve_experience'] = df_2020['Improve Experience'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))

empty = {'Negative': '', 'Positive': '', 
         "Improve Experience": ''}
df_2020.fillna(value=empty, inplace=True)

df_groupedreviews2020 = df_2020.groupby('Course Title')[['Negative', 'Positive', "Improve Experience"]].agg(' '.join)

positive_df_2020 = expand_comments(df_2020[['Department 1', 'Course Title', 'Negative', 'Positive', 'Improve Experience']], 
                              ['Department 1', 'Course Title'], 'Positive')
negative_df_2020 = expand_comments(df_2020[['Department 1','Course Title', 'Negative', 'Positive', 'Improve Experience']], 
                              ['Department 1', 'Course Title'], 'Negative')
improve_df_2020 = expand_comments(df_2020[['Department 1','Course Title', 'Negative', 'Positive', 'Improve Experience']], 
                              ['Department 1', 'Course Title'], 'Improve Experience')
positive_df_2020 = positive_df_2020[~positive_df_2020.Positive.str.contains('No Response Entered')]
negative_df_2020 = negative_df_2020[~negative_df_2020.Negative.str.contains('No Response Entered')]
improve_df_2020 = improve_df_2020[~improve_df_2020['Improve Experience'].str.contains('No Response Entered')]

# Input by list
positive_list_20 = positive_df_2020['Positive'].values.tolist()
negative_list_20 = negative_df_2020['Negative'].values.tolist()
improve_experience_list_20 = improve_df_2020['Improve Experience'].values.tolist()
list_by_sentiment_2020 = [positive_list_20, negative_list_20, improve_experience_list_20]

# Input by string
positive_text_20 = ' '.join(positive_df_2020['Positive'])
negative_text_20 = ' '.join(negative_df_2020['Negative'])
improve_experience_text_20 = ' '.join(improve_df_2020['Improve Experience'])
text_by_sentiment_2020 = [positive_text_20, negative_text_20, improve_experience_text_20]

# PyLDA Model
positive_lda_topics20 = get_lda_topics(positive_list_20, 3)
negative_lda_topics20 = get_lda_topics(negative_list_20, 7)
improve_lda_topics20 = get_lda_topics(improve_experience_list_20, 2)

#NMF Model
positive_nmf_topics20 = get_nmf_topics(positive_list_20, 3)
negative_nmf_topics20 = get_nmf_topics(negative_list_20, 7)
improve_nmf_topics20 = get_nmf_topics(improve_experience_list_20, 2)

#Luhn and LexRank Extractive Summary Models
positive_extractive_summaries_20 = get_extractive_summaries(positive_text_20)
negative_extractive_summaries_20 = get_extractive_summaries(negative_text_20)
improve_extractive_summaries_20 = get_extractive_summaries(improve_experience_text_20)

# Get the dominant topic and its probability for each comment
dominant_topics, dominant_probabilities = get_dominant_topic(positive_list_20, 3)

# Add the dominant topic column to the dataset
positive_df_2020['Positive Dominant Topic'] = dominant_topics
positive_df_2020['Positive Dominant Topic Probability'] = dominant_probabilities
positive_df_2020['Positive Dominant Topic Probability'] = round(positive_df_2020['Positive Dominant Topic Probability'], 2)
positive_df_2020['Positive Dominant Topic Probability'] = positive_df_2020['Positive Dominant Topic Probability'].astype(str)

# Dataset example
row_info_2020 = f"This dataset has {positive_df_2020.shape[0]} entries."

# Get the dominant topic and its probability for each comment
dominant_topics, dominant_probabilities = get_dominant_topic(negative_list_20, 7)

# Add the dominant topic column to the dataset
negative_df_2020['Negative Dominant Topic'] = dominant_topics
negative_df_2020['Negative Dominant Topic Probability'] = dominant_probabilities
negative_df_2020['Negative Dominant Topic Probability'] = round(negative_df_2020['Negative Dominant Topic Probability'], 2)
negative_df_2020['Negative Dominant Topic Probability'] = negative_df_2020['Negative Dominant Topic Probability'].astype(str)

themes_2020 = [
        "Course", "Content", "Teaching & Learning Set-Up"
]

themes_2020_negative = [
        "Course & Content", "Teaching & Learning Set-Up", "Department"
]

themes_data_2020 = [
    {
        # Values for Positive Themes
        "values": [35.9,21.9,42.2],
        "labels": themes_2020
    },
    {
        # Values for Negative Themes
        "values": [46.9,40.4,12.6],
        "labels": themes_2020_negative
    }
]

page7 = """
# Evaluating Learning For the 2019/20 Academic Year
#### Themes Overview
<|{themes_data_2020}|chart|type=pie|x[1]=0/values|x[2]=1/values|options={options_piechart_thematic}|layout={layout_piechart_thematic}|>

#### Strengths

##### Strength 1: Teaching & Learning Set-Up (42.2%)
<|{diagnosis_2020_positive.loc[:0]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2020[positive_df_2020['Positive Dominant Topic']==3])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2020[positive_df_2020['Positive Dominant Topic'].isin([3])]}|table|> 
|>

##### Strength 2: Course (35.9%)
<|{diagnosis_2020_positive.loc[1:1]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2020[positive_df_2020['Positive Dominant Topic']==1])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2020[positive_df_2020['Positive Dominant Topic'].isin([1])]}|table|> 
|>

##### Strength 3: Content (21.9%)
<|{diagnosis_2020_positive.loc[2:2]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2020[positive_df_2020['Positive Dominant Topic']==2])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2020[positive_df_2020['Positive Dominant Topic'].isin([2])]}|table|> 
|>

#### Areas for Improvement (AFI)

##### AFI 1: Course & Content (46.9%)
<|{diagnosis_2020_negative.loc[[0,2,4]]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2020[negative_df_2020['Negative Dominant Topic'].isin([6,3,1])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2020[negative_df_2020['Negative Dominant Topic'].isin([6,3,1])]}|table|> 
|>

##### AFI 2: Teaching & Learning Set-Up (40.4%)
<|{diagnosis_2020_negative.loc[[1,3,6]]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2020[negative_df_2020['Negative Dominant Topic'].isin([2,4,7])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2020[negative_df_2020['Negative Dominant Topic'].isin([2,4,7])]}|table|> 
|>

##### AFI 3: Department (12.6%)
<|{diagnosis_2020_negative.loc[5:5]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2020[negative_df_2020['Negative Dominant Topic'].isin([5])])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2020[negative_df_2020['Negative Dominant Topic'].isin([5])]}|table|> 
|>




"""



# 2020/21 Page
df_21 = pd.read_excel("input/NSS - National Student Survey/2021/NSS 2021 Comments - Blended Learning - Statistics.xlsx", 
                        header=[2], sheet_name=None)
df_21.keys()
df_2021 = df_21['NS2021 Comments by Department'].copy()
df_2021.rename(columns={"Positive comment": "Positive", "Negative comment": 'Negative',
                        "Institution own comment": "Blended Learning Review"}, inplace=True)
df_2021['non_tokenised_negative'] = df_2021['Negative'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
df_2021['non_tokenised_positive'] = df_2021['Positive'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
df_2021['non_tokenised_improve_experience'] = df_2021['Blended Learning Review'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
empty = {'Negative': '', 'Positive': '', 
         "Blended Learning Review": ''}
df_2021.fillna(value=empty, inplace=True)
df_groupedreviews2021 = df_2021.groupby('Course title')[['Negative', 'Positive', "Blended Learning Review"]].agg(' '.join)


positive_df_2021 = expand_comments(df_2021[['Dept name', 'Course title', 'Negative', 'Positive', 'Blended Learning Review']], 
                              ['Dept name', 'Course title'], 'Positive')
negative_df_2021 = expand_comments(df_2021[['Dept name','Course title', 'Negative', 'Positive', 'Blended Learning Review']], 
                              ['Dept name', 'Course title'], 'Negative')
improve_df_2021 = expand_comments(df_2021[['Dept name','Course title', 'Negative', 'Positive', 'Blended Learning Review']], 
                              ['Dept name', 'Course title'], 'Blended Learning Review')
positive_df_2021 = positive_df_2021[~positive_df_2021.Positive.str.contains('No Response Entered')]
negative_df_2021 = negative_df_2021[~negative_df_2021.Negative.str.contains('No Response Entered')]
improve_df_2021 = improve_df_2021[~improve_df_2021['Blended Learning Review'].str.contains('No Response Entered')]

# Input by list
positive_list_21 = positive_df_2021['Positive'].values.tolist()
negative_list_21 = negative_df_2021['Negative'].values.tolist()
improve_experience_list_21 = improve_df_2021['Blended Learning Review'].values.tolist()
list_by_sentiment_2021 = [positive_list_21, negative_list_21, improve_experience_list_21]

# Input by string
positive_text_21 = ' '.join(positive_df_2021['Positive'])
negative_text_21 = ' '.join(negative_df_2021['Negative'])
improve_experience_text_21 = ' '.join(improve_df_2021['Blended Learning Review'])
text_by_sentiment_2021 = [positive_text_21, negative_text_21, improve_experience_text_21]

# PyLDA Model
positive_lda_topics21 = get_lda_topics(positive_list_21, 3)
negative_lda_topics21 = get_lda_topics(negative_list_21, 4)
improve_lda_topics21 = get_lda_topics(improve_experience_list_21, 2)

#NMF Model
positive_nmf_topics21 = get_nmf_topics(positive_list_21, 3)
negative_nmf_topics21 = get_nmf_topics(negative_list_21, 4)
improve_nmf_topics21 = get_nmf_topics(improve_experience_list_21, 2)

#Luhn and LexRank Extractive Summary Models
positive_extractive_summaries_21 = get_extractive_summaries(positive_text_21)
negative_extractive_summaries_21 = get_extractive_summaries(negative_text_21)
improve_extractive_summaries_21 = get_extractive_summaries(improve_experience_text_21)

# Get the dominant topic and its probability for each comment
dominant_topics, dominant_probabilities = get_dominant_topic(positive_list_21, 3)

# Add the dominant topic column to the dataset
positive_df_2021['Positive Dominant Topic'] = dominant_topics
positive_df_2021['Positive Dominant Topic Probability'] = dominant_probabilities
positive_df_2021['Positive Dominant Topic Probability'] = round(positive_df_2021['Positive Dominant Topic Probability'], 2)
positive_df_2021['Positive Dominant Topic Probability'] = positive_df_2021['Positive Dominant Topic Probability'].astype(str)

# Dataset example
row_info_2021 = f"This dataset has {positive_df_2021.shape[0]} entries."

# Get the dominant topic and its probability for each comment
dominant_topics, dominant_probabilities = get_dominant_topic(negative_list_21, 4)

# Add the dominant topic column to the dataset
negative_df_2021['Negative Dominant Topic'] = dominant_topics
negative_df_2021['Negative Dominant Topic Probability'] = dominant_probabilities
negative_df_2021['Negative Dominant Topic Probability'] = round(negative_df_2021['Negative Dominant Topic Probability'], 2)
negative_df_2021['Negative Dominant Topic Probability'] = negative_df_2021['Negative Dominant Topic Probability'].astype(str)

themes_2021 = [
        "Community & Department", "Course & Content", "Teaching & Learning Set-Up"
]

themes_2021_negative = [
        "Community", "Course & Content", "Teaching & Learning Set-Up", "Department"
]

themes_data_2021 = [
    {
        # Values for Positive Themes
        "values": [35.5,38.4,26.1],
        "labels": themes_2021
    },
    {
        # Values for Negative Themes
        "values": [37.4,21.1,25,16.6],
        "labels": themes_2021_negative
    }
]

page8=  """
# Evaluating Learning For the 2020/21 Academic Year
#### Themes Overview
<|{themes_data_2021}|chart|type=pie|x[1]=0/values|x[2]=1/values|options={options_piechart_thematic}|layout={layout_piechart_thematic}|>

#### Strengths

##### Strength 1: Course & Content (38.4%)
<|{diagnosis_2021_positive.loc[:0]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2021[positive_df_2021['Positive Dominant Topic']==3])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2021[positive_df_2021['Positive Dominant Topic'].isin([3])]}|table|> 
|>

##### Strength 2: Community & Department (35.5%)
<|{diagnosis_2021_positive.loc[1:1]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2021[positive_df_2021['Positive Dominant Topic']==1])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2021[positive_df_2021['Positive Dominant Topic'].isin([1])]}|table|> 
|>

##### Strength 3: Teaching & Learning Set-Up (26.1%)
<|{diagnosis_2021_positive.loc[2:2]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2021[positive_df_2021['Positive Dominant Topic']==2])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2021[positive_df_2021['Positive Dominant Topic'].isin([2])]}|table|> 
|>

#### Areas for Improvement (AFI)

##### AFI 1: Community (37.4%)
<|{diagnosis_2021_negative.loc[:0]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2021[negative_df_2021['Negative Dominant Topic']==3])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2021[negative_df_2021['Negative Dominant Topic'].isin([3])]}|table|> 
|>

##### AFI 2: Teaching & Learning Set-Up (25%)
<|{diagnosis_2021_negative.loc[1:1]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2021[negative_df_2021['Negative Dominant Topic']==1])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2021[negative_df_2021['Negative Dominant Topic'].isin([1])]}|table|> 
|>

##### AFI 3: Course & Content (21.1%))
<|{diagnosis_2021_negative.loc[2:2]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2021[negative_df_2021['Negative Dominant Topic']==2])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2021[negative_df_2021['Negative Dominant Topic'].isin([2])]}|table|> 
|>

##### AFI 4: Department (16.6%)
<|{diagnosis_2021_negative.loc[3:3]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2021[negative_df_2021['Negative Dominant Topic']==4])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2021[negative_df_2021['Negative Dominant Topic'].isin([4])]}|table|> 
|>


"""

# 2022 Page
df_22 = pd.read_excel("input/NSS - National Student Survey/2022/NSS 2022 Comments ST Statistics v2.xlsx", 
                        header=[0], sheet_name=None)

df_22.keys()

df_2022 = df_22['NSS 2022 Comments by Department']

df_2022.rename(columns={"Positive comment": "Positive", "Negative comment": 'Negative', 
                        "Institution own comment: 'What teaching methods or activities did you find most beneficial, and why?'": "Beneficial Teaching Methods"}, inplace=True)

df_2022['non_tokenised_negative'] = df_2022['Negative'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
df_2022['non_tokenised_positive'] = df_2022['Positive'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
df_2022['non_tokenised_improve_experience'] = df_2022['Beneficial Teaching Methods'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))

empty = {'Negative': '', 'Positive': '', 
         "Beneficial Teaching Methods": ''}
df_2022.fillna(value=empty, inplace=True)

df_groupedreviews2022 = df_2022.groupby('Course Title')[['Negative', 'Positive', "Beneficial Teaching Methods"]].agg(' '.join)

df_2022_present = df_2022[['Department', 'Department Code', 'Course Code', 'Course Title', 
                                            'Beneficial Teaching Methods', 'Negative', 'Positive']]


positive_df_2022 = expand_comments(df_2022[['Department', 'Course Title', 'Negative', 'Positive', 'Beneficial Teaching Methods']], 
                              ['Department', 'Course Title'], 'Positive')
negative_df_2022 = expand_comments(df_2022[['Department','Course Title', 'Negative', 'Positive', 'Beneficial Teaching Methods']], 
                              ['Department', 'Course Title'], 'Negative')
improve_df_2022 = expand_comments(df_2022[['Department','Course Title', 'Negative', 'Positive', 'Beneficial Teaching Methods']], 
                              ['Department', 'Course Title'], 'Beneficial Teaching Methods')
positive_df_2022 = positive_df_2022[~positive_df_2022.Positive.str.contains('No Response Entered')]
negative_df_2022 = negative_df_2022[~negative_df_2022.Negative.str.contains('No Response Entered')]
improve_df_2022 = improve_df_2022[~improve_df_2022['Beneficial Teaching Methods'].str.contains('No Response Entered')]

# Input by list
positive_list_22 = positive_df_2022['Positive'].values.tolist()
negative_list_22 = negative_df_2022['Negative'].values.tolist()
beneficial_teaching_methods_list_22 = improve_df_2022['Beneficial Teaching Methods'].values.tolist()
list_by_sentiment_2022 = [positive_list_22, negative_list_22, beneficial_teaching_methods_list_22]

# Input by string
positive_text_22 = ' '.join(positive_df_2022['Positive'])
negative_text_22 = ' '.join(negative_df_2022['Negative'])
beneficial_teaching_methods_text_22 = ' '.join(improve_df_2022['Beneficial Teaching Methods'])
text_by_sentiment_2022 = [positive_text_22, negative_text_22, beneficial_teaching_methods_text_22]   

# PyLDA Model
positive_lda_topics22 = get_lda_topics(positive_list_22, 3)
negative_lda_topics22 = get_lda_topics(negative_list_22, 4)
improve_lda_topics22 = get_lda_topics(beneficial_teaching_methods_list_22, 15)

#NMF Model
positive_nmf_topics22 = get_nmf_topics(positive_list_22, 3)
negative_nmf_topics22 = get_nmf_topics(negative_list_22, 4)
improve_nmf_topics22 = get_nmf_topics(beneficial_teaching_methods_list_22, 15)

#Luhn and LexRank Extractive Summary Models
positive_extractive_summaries_22 = get_extractive_summaries(positive_text_22)
negative_extractive_summaries_22 = get_extractive_summaries(negative_text_22)
improve_extractive_summaries_22 = get_extractive_summaries(beneficial_teaching_methods_text_22)

# Get the dominant topic and its probability for each comment
dominant_topics, dominant_probabilities = get_dominant_topic(positive_list_22, 3)

# Add the dominant topic column to the dataset
positive_df_2022['Positive Dominant Topic'] = dominant_topics
positive_df_2022['Positive Dominant Topic Probability'] = dominant_probabilities
positive_df_2022['Positive Dominant Topic Probability'] = round(positive_df_2022['Positive Dominant Topic Probability'], 2)
positive_df_2022['Positive Dominant Topic Probability'] = positive_df_2022['Positive Dominant Topic Probability'].astype(str)

# Dataset example
row_info_2022 = f"This dataset has {positive_df_2022.shape[0]} entries."

# Get the dominant topic and its probability for each comment
dominant_topics, dominant_probabilities = get_dominant_topic(negative_list_22, 4)

# Add the dominant topic column to the dataset
negative_df_2022['Negative Dominant Topic'] = dominant_topics
negative_df_2022['Negative Dominant Topic Probability'] = dominant_probabilities
negative_df_2022['Negative Dominant Topic Probability'] = round(negative_df_2022['Negative Dominant Topic Probability'], 2)
negative_df_2022['Negative Dominant Topic Probability'] = negative_df_2022['Negative Dominant Topic Probability'].astype(str)

themes_2022 = [
        "Community & Department", "Course & Content", "Teaching & Learning Set-Up"
]

themes_2022_negative = [
        "Community & Department", "Teaching & Learning Set-Up", "Assessment & Feedback", "(Online) Teaching & Exams"
]

themes_data_2022 = [
    {
        # Values for Positive Themes
        "values": [50.1,15,34.9],
        "labels": themes_2022
    },
    {
        # Values for Negative Themes
        "values": [26.2, 17, 28.8, 28],
        "labels": themes_2022_negative
    }
]

page9 = """
# Evaluating Learning For the 2021/22 Academic Year
#### Themes Overview
<|{themes_data_2022}|chart|type=pie|x[1]=0/values|x[2]=1/values|options={options_piechart_thematic}|layout={layout_piechart_thematic}|>

#### Strengths

##### Strength 1: Community & Department (50.1%)
<|{diagnosis_2022_positive.loc[:0]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2022[positive_df_2022['Positive Dominant Topic']==3])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2022[positive_df_2022['Positive Dominant Topic'].isin([3])]}|table|> 
|>

##### Strength 2: Teaching & Learning Set-Up (34.9%)
<|{diagnosis_2022_positive.loc[1:1]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2022[positive_df_2022['Positive Dominant Topic']==1])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2022[positive_df_2022['Positive Dominant Topic'].isin([1])]}|table|> 
|>

##### Strength 3: Course & Content (15%)
<|{diagnosis_2022_positive.loc[2:2]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2022[positive_df_2022['Positive Dominant Topic']==2])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2022[positive_df_2022['Positive Dominant Topic'].isin([2])]}|table|> 
|>

#### Areas for Improvement (AFI)

##### AFI 1: Assessment & Feedback (28.8%)
<|{diagnosis_2022_negative.loc[:0]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2022[negative_df_2022['Negative Dominant Topic']==1])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2022[negative_df_2022['Negative Dominant Topic'].isin([1])]}|table|> 
|>

##### AFI 2: (Online) Teaching & Exams (28%)
<|{diagnosis_2022_negative.loc[1:1]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2022[negative_df_2022['Negative Dominant Topic']==3])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2022[negative_df_2022['Negative Dominant Topic'].isin([3])]}|table|> 
|>

##### AFI 3: Community & Department (26.2%))
<|{diagnosis_2022_negative.loc[2:2]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2022[negative_df_2022['Negative Dominant Topic']==4])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2022[negative_df_2022['Negative Dominant Topic'].isin([4])]}|table|> 
|>

##### AFI 4: Teaching & Learning Set-Up (17%)
<|{diagnosis_2022_negative.loc[3:3]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2022[negative_df_2022['Negative Dominant Topic']==2])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2022[negative_df_2022['Negative Dominant Topic'].isin([2])]}|table|> 
|>


"""



df_23 = pd.read_excel("input/NSS - National Student Survey/2023/2023-08-17_NSS23_Comments_ST Statistics.xlsx", 
                        header=[0], sheet_name=None)


df_2023 = df_23['Comments by Dept_ST']

df_2023.rename(columns={"Positive comment": "Positive", "Negative comment": 'Negative',                       "Institution own comment: 'What is the one thing we could have done to improve your overall experience?'": "Improve Experience"}, inplace=True)

df_2023['non_tokenised_negative'] = df_2023['Negative'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
df_2023['non_tokenised_positive'] = df_2023['Positive'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))
df_2023['non_tokenised_improve_experience'] = df_2023['Improve Experience'].apply(Preprocessor.text_preprocessor, args=('lemmatisation', False))

empty = {'Negative': '', 'Positive': '', 
         "Improve Experience": ''}
df_2023.fillna(value=empty, inplace=True)

df_groupedreviews2023 = df_2023.groupby('Course Title')[['Negative', 'Positive', "Improve Experience"]].agg(' '.join)

# Preview Dataset Input
df_2023_present = df_2023[['Department Name', 'Department Code', 'Course Code', 'Course Title', 
                                            'Improve Experience', 'Negative', 'Positive']]

# Input by list
positive_df_2023 = expand_comments(df_2023[['Department Name', 'Course Title', 'Negative', 'Positive', 'Improve Experience']], 
                              ['Department Name', 'Course Title'], 'Positive')
negative_df_2023 = expand_comments(df_2023[['Department Name','Course Title', 'Negative', 'Positive', 'Improve Experience']], 
                              ['Department Name', 'Course Title'], 'Negative')
improve_df_2023 = expand_comments(df_2023[['Department Name','Course Title', 'Negative', 'Positive', 'Improve Experience']], 
                              ['Department Name', 'Course Title'], 'Improve Experience')
positive_df_2023 = positive_df_2023[~positive_df_2023.Positive.str.contains('No Response Entered')]
negative_df_2023 = negative_df_2023[~negative_df_2023.Negative.str.contains('No Response Entered')]
improve_df_2023 = improve_df_2023[~improve_df_2023['Improve Experience'].str.contains('No Response Entered')]

# Input by list
positive_list_23 = positive_df_2023['Positive'].values.tolist()
negative_list_23 = negative_df_2023['Negative'].values.tolist()
improve_experience_list_23 = improve_df_2023['Improve Experience'].values.tolist()
list_by_sentiment_2023 = [positive_list_23, negative_list_23, improve_experience_list_23]

# Input by string
positive_text_23 = ' '.join(positive_df_2023['Positive'])
negative_text_23 = ' '.join(negative_df_2023['Negative'])
improve_experience_text_23 = ' '.join(improve_df_2023['Improve Experience'])
text_by_sentiment_2023 = [positive_text_23, negative_text_23, improve_experience_text_23]

# PyLDA Model
positive_lda_topics23 = get_lda_topics(positive_list_23, 4)
negative_lda_topics23 = get_lda_topics(negative_list_23, 5)
# improve_lda_topics23 = get_lda_topics(improve_experience_list_23, 15)

#NMF Model
positive_nmf_topics23 = get_nmf_topics(positive_list_23, 4)
negative_nmf_topics23 = get_nmf_topics(negative_list_23, 5)
# improve_nmf_topics23 = get_nmf_topics(improve_experience_list_23, 15)

#Luhn and LexRank Extractive Summary Models
positive_extractive_summaries_23 = get_extractive_summaries(positive_text_23)
negative_extractive_summaries_23 = get_extractive_summaries(negative_text_23)
improve_extractive_summaries_23 = get_extractive_summaries(improve_experience_text_23)

# Get the dominant topic and its probability for each Positive comment
dominant_topics, dominant_probabilities = get_dominant_topic(positive_list_23, 4)
# Add the dominant topic column to the dataset
positive_df_2023['Positive Dominant Topic'] = dominant_topics
positive_df_2023['Positive Dominant Topic Probability'] = dominant_probabilities
positive_df_2023['Positive Dominant Topic Probability'] = round(positive_df_2023['Positive Dominant Topic Probability'], 2)
positive_df_2023['Positive Dominant Topic Probability'] = positive_df_2023['Positive Dominant Topic Probability'].astype(str)

# Get the dominant topic and its probability for each Negative comment
dominant_topics, dominant_probabilities = get_dominant_topic(negative_list_23, 5)
# Add the dominant topic column to the dataset
negative_df_2023['Negative Dominant Topic'] = dominant_topics
negative_df_2023['Negative Dominant Topic Probability'] = dominant_probabilities
negative_df_2023['Negative Dominant Topic Probability'] = round(negative_df_2023['Negative Dominant Topic Probability'], 2)
negative_df_2023['Negative Dominant Topic Probability'] = negative_df_2023['Negative Dominant Topic Probability'].astype(str)

# Dataset example
row_info_2023 = f"This dataset has {positive_df_2023.shape[0]} entries."

themes_2023 = [
        "Community & Department", "Course & Content", "Teaching & Learning Set-Up"
]

themes_2023_negative = [
        "Community", "Course & Content", "Teaching & Learning Set-Up", "Department", "Learning Inconveniences"
]

themes_data_2023 = [
    {
        # Values for Positive Themes
        "values": [36.8,33.2,30],
        "labels": themes_2023
    },
    {
        # Values for Negative Themes
        "values": [13.9,25.3,31.2,14,15.6],
        "labels": themes_2023_negative
    }
]

page10 = """
# Evaluating Learning For the 2022/23 Academic Year
#### Themes Overview
<|{themes_data_2023}|chart|type=pie|x[1]=0/values|x[2]=1/values|options={options_piechart_thematic}|layout={layout_piechart_thematic}|>

#### Strengths

##### Strength 1: Community & Department (25.4%)
<|{diagnosis_2023_positive.loc[:0]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2023[positive_df_2023['Positive Dominant Topic']==1])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2023[positive_df_2023['Positive Dominant Topic'].isin([1])]}|table|> 
|>


##### Strength 2: Course & Content (21.7%)
<|{diagnosis_2023_positive.loc[1:1]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2023[positive_df_2023['Positive Dominant Topic']==2])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2023[positive_df_2023['Positive Dominant Topic'].isin([2])]}|table|> 
|>

##### Strength 3: Teaching & Learning Set-Up (18.5%)
<|{diagnosis_2023_positive.loc[2:2]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2023[positive_df_2023['Positive Dominant Topic']==3])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2023[positive_df_2023['Positive Dominant Topic'].isin([3])]}|table|> 
|>


##### Strength Other: Multiple Themes (34.4%)
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((positive_df_2023[positive_df_2023['Positive Dominant Topic']==4])['Positive']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{positive_df_2023[positive_df_2023['Positive Dominant Topic'].isin([4])]}|table|> 
|>

#### Areas for Improvement (AFI)

##### AFI 1: Teaching & Learning Set-Up (31.2%)
<|{diagnosis_2023_negative.loc[:0]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2023[negative_df_2023['Negative Dominant Topic']==1])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2023[negative_df_2023['Negative Dominant Topic'].isin([1])]}|table|> 
|>

##### AFI 2: Course & Content (25.3%)
<|{diagnosis_2023_negative.loc[1:1]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2023[negative_df_2023['Negative Dominant Topic']==5])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2023[negative_df_2023['Negative Dominant Topic'].isin([5])]}|table|> 
|>

##### AFI 3: Learning Inconveniences (15.6%)
<|{diagnosis_2023_negative.loc[2:2]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2023[negative_df_2023['Negative Dominant Topic']==3])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2023[negative_df_2023['Negative Dominant Topic'].isin([3])]}|table|> 
|>

##### AFI 4: Department (14%)
<|{diagnosis_2023_negative.loc[3:3]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2023[negative_df_2023['Negative Dominant Topic']==4])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2023[negative_df_2023['Negative Dominant Topic'].isin([4])]}|table|> 
|>

##### AFI 5: Community (13.9%)
<|{diagnosis_2023_negative.loc[4:4]}|table|> 
<|Summaries|expandable|expanded=False|
<|{get_extractive_summaries(' '.join((negative_df_2023[negative_df_2023['Negative Dominant Topic']==2])['Negative']))}|table|> 
|>
<|Reviews|expandable|expanded=False|
<|{negative_df_2023[negative_df_2023['Negative Dominant Topic'].isin([2])]}|table|> 
|>

"""




pages = {
    "/":"<|toggle|theme|>\n<center>\n<|navbar|>\n</center>",
    "Overview": page1,
    "2014/15": page2,
    "2015/16": page3,
    "2016/17": page4,
    "2017/18": page5,
    "2018/19": page6,
    "2019/20": page7,
    "2020/21": page8,
    "2021/22": page9,
    "2022/23": page10
}

if __name__ == "__main__":
    Gui(pages=pages).run(use_reloader=False)
