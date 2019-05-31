#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
from datetime import datetime
import joblib
import json
import logging
import numpy as np
import re
import s3io
from scipy.spatial import distance
import sys
import time

log = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)


credentials = dict(
    aws_access_key_id=None,
    aws_secret_access_key=None
)

# 1. Combined Model
model_effective_date = "2018-08-03"
model_uuid = 'e47c65bc-8603-493d-a07b-c5b77561fc86'
sample_model = True

if sample_model:
    combined_model_url = "candidate_classification_rf_model_with_w2v_vectors_VectorsSaved_sample-{}-{}.pkl".format(model_effective_date, model_uuid)
else:
    combined_model_url = "candidate_classification_rf_model_with_w2v_vectors_VectorsSaved-{}-{}.pkl".format(model_effective_date, model_uuid)


with open(combined_model_url) as f:
  combined_model = joblib.load(f)

senior_prefix = ["senior", "staff", "lead", "head", "principle", "principal", "managing", "level"]
junior_prefix = ["junior", "assistant", "associate", "intern", "self", "contract", "student", "freelance", "freelancer",
                 "internship", "graduate", "undergraduate"]

seniority = set(senior_prefix + junior_prefix)
allowed_seniority_suffix = {"lead", "assistant", "associate"}
blacklist = {"student", "graduate", "undergraduate", "candidate"}

pattern_seniority = re.compile(r'\b(' + r'|'.join(seniority) + r')\b')
pattern_blacklist = re.compile(r'\b(' + r'|'.join(blacklist) + r')\b')
pattern_roman_numerals = re.compile(r'\b(level )?([i0-9]+)\b')
pattern_number_prefix = re.compile(r'\b([0-9]+[a-z]* (grade)*)\b')


title_acronyms = [
  ('ceo', 'chief executive officer'),
  ('coo', 'chief operating officer'),
  ('cto', 'chief technology officer'),
  ('cfo', 'chief finance officer'),
  ('csr', 'customer service representative'),
  ('avp', 'assistant vice president'),
  ('svp', 'senior vice president'),
  ('vp', 'vice president'),
  ('swe', 'software engineer'),
  ('sdet', 'software development engineer in test'),
  ('ae', 'account executive'),
  ('eae', 'enterprise account executive'),
  ('tsr', 'technical support representative'),
  ('tse', 'technical support engineer'),
  ('rn', 'registered nurse'),
  ('lpn', 'licensed practical nurse'),
  ('lvn', 'licensed vocational nurse'),
  ('lpn', 'licensed practical nurse'),
  ('cmo', 'chief marketing officer'),
  ('cdo', 'chief data officer'),
  ('chro', 'chief human resources officer'),

  # not really acronyms but rather mispelling
  ('0wner', 'owner'),
  ('consutant', 'consultant'),
]

replace_list = [
  ('asst', 'assistant '),
  ('admin', 'administrator '),
  ('sr', 'senior '),
  ('jr', 'junior '),
  ('rep', 'representative '),
  ('mgr', 'manager '),
  ('mngr', 'manager '),
  ('mngt', 'management '),
  ('engr', 'engineer '),
  ('eng', 'engineer '),
  ('sys', 'system '),
  ('tech', 'technical '),
  ('gov', 'government'),
  ('govt', 'government'),
  ('drivers', 'driver '),
  ('hr', 'human resources'),
  ('qa', 'quality assurance'),
  ('sw', 'software'),
  ('med/surg', 'medical surgical'),
  ('ui', 'user interface'),
  ('ux', 'user experience'),
  ('seo', 'search engine optimization'),
  ('sem', 'search engine marketing'),
  ('js', 'javascript'),
  ('xd', 'experience design'),
  ('uxd', 'user experience design'),
  ('frontend', 'front end'),
  ('front-end', 'front end'),
  ('backend', 'back end'),
  ('back-end', 'back end'),
  ('fullstack', 'full stack'),
  ('full-stack', 'full stack'),
]

title_acronyms_dict = {k: v for k, v in title_acronyms}
replace_dict = {k: v for k, v in replace_list}

punct = '[!"#$%&\'*,.:;<=>?@\\^_`{|}()~]'
punct1 = '[—]'

pattern_punct = re.compile(punct)
pattern_punct1 = re.compile(punct1)
pattern_multispace = re.compile("\s+")
pattern_job_code = re.compile("#[a-z0-9-]{3,10}|[a-z]*[0-9-]{3,8}[a-z]*")
pattern_numbers = re.compile(r'\b[0-9]+\b')
pattern_unicode = re.compile(r'[^\x00-\x7F]+')
pattern_sq = re.compile(r'\[.*?\]')

rx_acronym = re.compile('\\b|\\b'.join(map(re.escape, title_acronyms_dict)))
rx_replace = re.compile('\\b|\\b'.join(map(re.escape, replace_dict)))


class CandidateApproval(object):
    MODELS = None
    PREDICTION_THRESHOLD = 0.55

    FEATURE_ORDER = [
        # 'reputedly_profile_id',
        # 'ar_job_id',
        'profile_experience_most_recent_has_internal_move_6m_ind',
        'profile_fast_riser_wgted',
        'profile_working_for_unicorn_company_wgted',
        'profile_is_entrepreneur_wgted',
        'profile_has_entrepreneurial_experience_wgted',
        'profile_in_high_demand_wgted',
        'profile_is_recent_vip_wgted',
        'profile_job_industry_match',
        'profile_job_industry_category_match',
        'profile_job_company_size_match',
        'profile_job_company_position_cluster_match',
        'profile_job_job_family_match',
        'profile_experience_seniority_progression_rate',
        'profile_matched_skill_set_pct',
        'profile_title_match_pct',
        'profile_job_TR_criteria_cosine_similarity',
        'profile_job_C_criteria_cosine_similarity',
        'profile_job_S_cosine_similarity',
        'profile_position_title_canonical_seniority_level_ord_most_recent_avgflr_normed',
        'profile_experience_total_tenure_gap_and_recency_adj_avg_normed',
        'profile_most_recent_position_end_days_normed',
        'relevant_yoe_diff_mean_normed',
        'total_yoe_diff_mean_normed',
        'category',
        'profile_TRCS_vector',
        'job_criteria_TRCS_vector',
    ]

    POSITIVE_FEATURES = OrderedDict(
        [
            ('profile_title_match_pct', 0.5),
            ('profile_job_TR_criteria_cosine_similarity', 0.6),
            ('profile_matched_skill_set_pct', 0.5),
            ('profile_job_S_cosine_similarity', 0.6),
            ('profile_job_C_criteria_cosine_similarity', 0.45),
            ('profile_job_company_position_cluster_match', 1),
            ('relevant_yoe_diff_mean_normed', 0),
            ('total_yoe_diff_mean_normed', 0),
            ('profile_job_industry_match', 1),
            ('profile_job_industry_category_match', 1),
            ('profile_job_company_size_match', 1),
            ('profile_job_job_family_match', 1),
            ('profile_in_high_demand_wgted', 1),
            ('profile_is_recent_vip_wgted', 1),
            ('profile_experience_seniority_progression_rate', 1e-05),
            ('profile_fast_riser_wgted', 1),
            ('profile_working_for_unicorn_company_wgted', 1)
        ]
    )

    NEGATIVE_FEATURES = OrderedDict(
        [
            ('profile_title_match_pct', 0.5),
            ('profile_job_TR_criteria_cosine_similarity', 0.6),
            ('profile_matched_skill_set_pct', 0.5),
            ('profile_job_S_cosine_similarity', 0.6),
            ('profile_job_C_criteria_cosine_similarity', 0.45),
            ('profile_job_company_position_cluster_match', 1),
            ('relevant_yoe_diff_mean_normed', 0),
            ('total_yoe_diff_mean_normed', 0),
            ('profile_job_industry_match', 1),
            ('profile_job_industry_category_match', 1),
            ('profile_job_company_size_match', 1),
            ('profile_job_job_family_match', 1),
        ]
    )

    QUALIFICATION_FEATURES = [
        'profile_job_job_family_match',
        'profile_matched_skill_set_pct',
        'profile_title_match_pct',
        'profile_job_TR_criteria_cosine_similarity',
        'profile_job_C_criteria_cosine_similarity',
        'profile_job_S_cosine_similarity'
    ]

    NORMALIZE_COLS = [
        'profile_position_title_canonical_seniority_level_ord_most_recent_avgflr',
        'profile_experience_total_tenure_gap_and_recency_adj_avg',
        'relevant_yoe_diff_min',
        'relevant_yoe_diff_max',
        'relevant_yoe_diff_mean',
        'total_yoe_diff_min',
        'total_yoe_diff_max',
        'total_yoe_diff_mean',
        'profile_most_recent_position_end_days'
    ]

    # # instead of dividing by the max of each column, which can go extreme, use "mean+3*std" for better result
    NORMALIZE_DICT = {
        'profile_position_title_canonical_seniority_level_ord_most_recent_avgflr': 4,
        'profile_experience_total_tenure_gap_and_recency_adj_avg': 8.2,
        'relevant_yoe_diff_min': 8.0,
        'relevant_yoe_diff_max': 24.0,
        'relevant_yoe_diff_mean': 16.0,
        'total_yoe_diff_min': 12.0,
        'total_yoe_diff_max': 25.0,
        'total_yoe_diff_mean': 19.0,
        'profile_most_recent_position_end_days': 170.0
    }

    def __init__(self, models=None, model_url=None, model_mode=None):
        CandidateApproval.load_models(models, model_url, model_mode)

    @classmethod
    def load_models(cls, models=None, model_url=None, model_mode=None):
        start = time.time()
        if cls.MODELS is not None:
            return

        if models:
            log.info("Loading supplied models...")
            cls.MODELS = models
        else:
            if model_mode == "s3":
                with s3io.open(model_url, mode='rb', **credentials) as f:
                    loaded_model = joblib.load(f)
            else:
                with open(model_url, mode='rb') as f:
                    loaded_model = joblib.load(f)

            if loaded_model is None:
                raise Exception("Model does not exist")

            cls.MODELS = loaded_model

        end = time.time()
        log.info("Loaded the models in %d seconds" % (end - start))

    @staticmethod
    def multiple_replace(text, adict, rx):
        def one_xlat(match):
            return adict[match.group(0)]

        return rx.sub(one_xlat, text)

    def clean_text(self, text):
        if not text:
            return ''

        text = text.replace("\x01", "").lower()

        text = pattern_numbers.sub(
            " ", pattern_unicode.sub(
                " ", pattern_punct1.sub(
                    " ", pattern_punct.sub(
                        " ", pattern_job_code.sub(
                            " ", pattern_sq.sub(
                                " ", text))))))

        text = self.multiple_replace(text, title_acronyms_dict, rx_acronym)  # Expand acronyms
        text = self.multiple_replace(text, replace_dict, rx_replace)  # Replace

        text = pattern_multispace.sub(" ", text).strip()

        return text

    @staticmethod
    def get_ngrams(text="", n_gram_min=2):
        ngrams = []

        word_list = text.split()
        n_gram_max = len(word_list)

        for n in xrange(n_gram_min, n_gram_max + 1):
            ngrams += [" ".join(word_list[w: w + n]) for w in xrange(0, len(word_list) - n + 1)]

        return ngrams

    @staticmethod
    def tag_text(text, prefix, delimiter="_"):
        if not text:
            return None
        text = text.replace(" ", delimiter)
        return "{}{}{}".format(prefix, delimiter, text)

    @staticmethod
    def tag_text_list(text_list, prefix, delimiter="_"):
        return_list = []

        if not text_list:
            return None

        for text in text_list:
            return_list.append("{}{}{}".format(prefix, delimiter, text.replace(" ", delimiter)))
        return return_list

    def canonicalize_title(self, title):
        """
        Split the title into ngrams. Return ngrams that are part of the top titles list
        """

        # Order
        # 1. Exact Match
        # 2. n-gram size
        # 3. n-gram profile count

        title = self.clean_text(title)

        if title in self.MODELS["canonical_titles"]:
            return title, self.MODELS["canonical_titles"][title][0], self.MODELS["canonical_titles"][title][1]

        title_ngrams = self.get_ngrams(title)

        top_ngrams = sorted(
            [(ng, self.MODELS["canonical_titles"][ng]) for ng in title_ngrams if ng in self.MODELS["canonical_titles"]],
            key=lambda x: (len(x[0].split()), x[2]), reverse=True)
        if top_ngrams:
            return top_ngrams[0][0], self.MODELS["canonical_titles"][top_ngrams[0][0]][0], \
                   self.MODELS["canonical_titles"][top_ngrams[0][0]][1]

        return None, None, None

    def get_title_role(self, title):
        if not title:
            return None

        title = ' '.join(title.split('_'))
        title = self.clean_text(title)

        if title.endswith(tuple(allowed_seniority_suffix)):
            return title
        elif re.search(pattern_blacklist, title):
            return None
        title = pattern_seniority.sub("", title)
        title = pattern_roman_numerals.sub("", title)
        title = pattern_number_prefix.sub("", title)
        title = title.strip()

        if title:
            return title

        return None

    def get_title_role_list(self, titles):
        roles = []
        for title in titles:
            title_role = self.get_title_role(title.replace("_", " "))
            if title_role:
                roles.append(title_role.replace(" ", "_"))
        return roles

    @staticmethod
    def get_attribute_from_criteria(criteria, attribute_name):
        if isinstance(criteria, str) or isinstance(criteria, unicode):
            criteria = json.loads(criteria)
        try:
            return criteria[attribute_name]
        except:
            return None

    @staticmethod
    def concat_string_lists(l1, l2):
        if not l1:
            l1 = []

        if not l2:
            l2 = []

        return list(set(l1).union(set(l2)))

    @staticmethod
    def str_to_list(my_string):
        if my_string is None:
            return None
        else:
            return my_string.split(" | ")

    def get_similar_terms(self, skills, top_n_similar=5):
        similar_skills = []
        if not skills:
            return []
        else:
            for skill in skills:
                if skill in self.MODELS["skill_similarity_dict"]:
                    similar_skills += [k for (k, v) in
                                       self.MODELS["skill_similarity_dict"].get(skill, []).items()[:top_n_similar]]
        return list(set(similar_skills).union(set(skills)))

    @staticmethod
    def check_skill_match(required_skills, preferred_skills, profile_skills):
        if not required_skills:
            required_skills = []

        if not preferred_skills:
            preferred_skills = []

        required_dict = {'preferred': len(preferred_skills), 'required': len(required_skills)}
        match_dict = {'preferred': 0, 'required': 0}

        if not required_skills and not preferred_skills and not profile_skills:
            return 0.25
        elif not profile_skills:
            return 0.0
        else:

            for skill in required_skills:
                if skill in profile_skills:
                    match_dict["required"] += 1

            for skill in preferred_skills:
                if skill in profile_skills:
                    match_dict["preferred"] += 1

        if required_dict['required'] > 0 and required_dict['preferred'] == 0:
            match_pct = match_dict['required'] * 1.0 / required_dict['required']
        elif required_dict['preferred'] > 0 and required_dict['required'] == 0:
            match_pct = match_dict['preferred'] * 1.0 / required_dict['preferred']
        elif required_dict['required'] > 0 and required_dict['preferred'] > 0:
            match_pct = (match_dict['required'] * 1.0 / required_dict['required']) * 0.7 + (match_dict['preferred'] * 1.0 / required_dict['preferred']) * 0.3
        else:
            match_pct = 0.0

        return match_pct

    @staticmethod
    def jaccard_distance(str1, str2):
        s1 = set(str1.split())
        s2 = set(str2.split())
        return len(s1 & s2) * 1.0 / len(s1 | s2)

    def check_match(self, required_entities, profile_entities):
        if not required_entities and not profile_entities:
            return 0.25

        elif profile_entities is None or len(profile_entities) == 0:
            return 0.0

        matched = []

        for entity in required_entities:
            jac = self.jaccard_distance(entity.replace("_", " "), profile_entities)

            if jac > 0:
                matched.append(jac)

        if matched:
            return max(matched)

        return 0.0

    @staticmethod
    def combine(title, role, company, skills):
        title = title if title else []
        role = role if role else []
        company = company if company else []
        if skills:
            skills = ['_'.join(skill.split()) for skill in skills]
        else:
            skills = []
        if isinstance(title, str) or isinstance(title, unicode):
            title = [title]
        if isinstance(role, str) or isinstance(role, unicode):
            role = [role]
        if isinstance(company, str) or isinstance(company, unicode):
            company = [company]
        if isinstance(skills, str) or isinstance(skills, unicode):
            skills = [skills]
        res = title + role + company + skills
        if len(res) > 0:
            return res

        return None

    def get_vectors(self, entity, industry="", company_size="", job_family=""):
        vector = np.array([0.0] * 100)
        cnt = 0

        if (isinstance(entity, str) or isinstance(entity, unicode)) and entity in self.MODELS["all_vectors"]:
            return self.MODELS["all_vectors"][entity]

        if isinstance(entity, list):
            for item in entity:
                if item in self.MODELS["all_vectors"]:
                    vector = np.add(vector, np.array(list(self.MODELS["all_vectors"][item])))

                    cnt += 1
            if cnt > 0:
                return vector / cnt

        if industry and company_size and (industry, company_size) in self.MODELS["industry_size_vectors"]:
            return self.MODELS["industry_size_vectors"][(industry, company_size)][0]

        if job_family and job_family in self.MODELS["job_family_vectors"]:
            return self.MODELS["job_family_vectors"][job_family][0]
        return vector

    @staticmethod
    def elementwise_vector_product(v1, v2):
        try:
            return np.sum(np.array(list(v1) * np.array(list(v2))))
        except ValueError:
            return None

    @staticmethod
    def vector_cosine_similarity(v1, v2):
        default_value = 0.25
        try:
            d = 1.0 - float(distance.cosine(np.array(list(v1)), np.array(list(v2))))
            if d >= 0 or d <= 1:
                return d
            return default_value
        except ValueError:
            return default_value

    @staticmethod
    def pct_contains(profiles, jobs):
        if profiles is None or len(profiles) == 0 or jobs is None or len(jobs) == 0:
            return 0.0

        t1 = set(profiles)
        t2 = set(jobs)
        return len(t1.intersection(t2)) * 100.0 / len(t2)

    @staticmethod
    def get_entities_from_criteria(criteria, entity_type, key, option=""):
        return_entities = []

        try:
            if isinstance(criteria, str) or isinstance(criteria, unicode):
                criteria = json.loads(criteria)

            entities = criteria.get(entity_type)

            for entity in entities:
                search_option = entity.get("search_option", "")

                if option == search_option or not search_option or not option:
                    return_entities.append(entity[key].lower().replace(" ", "_"))
            return return_entities
        except:
            return None

    @staticmethod
    def check_qualifications_match(profile):
        all_match_lst = [1 if profile[k] == 0.25 else 0 for k in CandidateApproval.QUALIFICATION_FEATURES]
        return sum(all_match_lst)

    @classmethod
    def validate_prediction(cls, predictions):
        total = len(predictions)
        approved = len([1 for p in predictions if p["prediction_probability"] >= CandidateApproval.PREDICTION_THRESHOLD])
        percent_approved = approved * 100.0 / total

        if (approved < 20 or percent_approved < 20) and total > 20:
            return False

        return True

    def predict(self, model_features):
        prediction_results = []
        for candidate_feature_lookup in model_features:
            candidate_features = []
            positive_features = []
            negative_features = []

            feature_values = []

            for feature_name in CandidateApproval.FEATURE_ORDER:
                if feature_name not in candidate_feature_lookup:
                    raise Exception("FEATURE DOES NOT EXIST")

                feat = candidate_feature_lookup[feature_name]

                if feature_name in CandidateApproval.POSITIVE_FEATURES and feat >= CandidateApproval.POSITIVE_FEATURES[feature_name]:
                    positive_features.append(feature_name)

                if feature_name in CandidateApproval.NEGATIVE_FEATURES and feat < CandidateApproval.NEGATIVE_FEATURES[feature_name]:
                    negative_features.append(feature_name)

                if feature_name.endswith("_vector"):
                    feat = feat.tolist()
                else:
                    feature_values.append((feature_name, feat))

                if feature_name == "category":
                    category = feat if feat in CandidateApproval.MODELS["categories"] else "uncategorized"
                    tmp_cat_list = [0] * len(CandidateApproval.MODELS["categories"])

                    ind = CandidateApproval.MODELS["categories"].index(category)

                    if ind >= 0:
                        tmp_cat_list[ind] = 1

                    feat = tmp_cat_list

                if isinstance(feat, list):
                    candidate_features.extend(feat)
                else:
                    candidate_features.extend([feat])

            prediction_probability = self.MODELS["classifier"].predict_proba(np.array([candidate_features]))[0][1]
            pred = prediction_probability > CandidateApproval.PREDICTION_THRESHOLD and set(positive_features).intersection(set(CandidateApproval.QUALIFICATION_FEATURES))

            prediction_result = {
                "prediction_probability": prediction_probability,
                "prediction": 'approved' if pred else 'rejected',
                "top_features": sorted(positive_features, key=lambda x: CandidateApproval.POSITIVE_FEATURES.keys().index(x)) if pred else sorted(negative_features, key=lambda x: CandidateApproval.NEGATIVE_FEATURES.keys().index(x)),
                "feature_values": feature_values
            }

            if candidate_feature_lookup["profile_all_qualifications_match_score"] >= 4:
                prediction_result["prediction"] = "not enough information"
                prediction_result["top_features"] = []

            prediction_results.append(prediction_result)

        prediction_valid = CandidateApproval.validate_prediction(prediction_results)

        prediction_results_updated = []
        if not prediction_valid:
            for p in prediction_results:
                p["prediction"] = "NA"
                prediction_results_updated.append(p)
        else:
            prediction_results_updated = prediction_results

        return prediction_results_updated


def main():
    c = CandidateApproval(models=None,
                          model_url=combined_model_url,
                          model_mode="local")

    sample_good_match = {
        'ar_candidate_id': 102805174,
        'ar_job_id': 1381,
        'ar_job_title': 'registered nurse',
        'ar_job_org_id': 3549,
        'category': 'nursing',
        'job_company_name_canonical': 'st lukes',
        'job_company_size_derived': '201-500',
        'job_company_industry_derived': 'hospital and health care',
        'job_company_industry_category_derived': 'medical',
        'job_company_position_cluster_id': 177,
        'job_company_specialty_cluster_id': None,
        'criteria': '{"advanced_degree":false,"companies":null,"company_sizes":null,"countries":null,"diversity":true,"fields_of_study":null,"four_year_degree":false,"industries":null,"job_titles":[{"title":"rn","search_option":"preferred"},{"title":"registered nurse registered nurse","search_option":"preferred"}],"locations":[{"location":"Kansas City, Missouri","radius":50},{"location":"Chillicothe, Missouri","radius":50}],"management_experience":false,"not_job_hopper":false,"only_current_company_size":false,"schools":null,"skills":[{"name":"nursing","search_option":"preferred"},{"name":"acute care","search_option":"preferred"},{"name":"registered nurses","search_option":"preferred"},{"name":"bls","search_option":"preferred"}],"years_of_relevant_experience_max":15,"years_of_relevant_experience_min":2,"years_of_relevant_experience":1,"years_of_total_experience":0,"additional_notes":"200 mile radius of Kansas City","years_of_experience_max":25,"years_of_experience_min":1,"category":""}'.lower(),
        'profile_skill_desc': 'supervisory skills | pals | leadership development | medicine | outlook | community outreach | microsoft word | behavioral health | customer service | emr | clinical research | ehr | managed care | healthcare consulting | teaching | physician relations | treatment | training | team building | inpatient | coaching | public health | healthcare information technology | policy | medical billing | healthcare management | case managment | nonprofits | nutrition | hipaa | fundraising | human resources | cpr certified | medicare | nursing | hospitals | medical terminology | budgets | organizational development | public speaking | bls | microsoft office | mental health | executive coaching | non-profits | strategic planning | software documentation | healthcare | powerpoint | grant writing'.lower(),
        'profile_position_company_name_canonical': 'kansas city va medical center',
        'profile_company_size_derived': '10001+',
        'profile_company_industry_derived': 'government administration',
        'profile_company_industry_category_derived': 'public',
        'profile_company_position_cluster_id': None,
        'profile_company_specialty_cluster_id': None,
        'profile_position_most_recent_role': 'registered nurse',
        'profile_experience_most_recent_has_internal_move_6m_ind': 0,
        'profile_fast_riser_wgted': 0,
        'profile_working_for_unicorn_company_wgted': 0,
        'profile_is_entrepreneur_wgted': 0,
        'profile_has_entrepreneurial_experience_wgted': 0,
        'profile_in_high_demand_wgted': 0,
        'profile_is_recent_vip_wgted': 0,
        'profile_position_title_canonical_seniority_level_ord_most_recent_avgflr': 2.0,
        'profile_experience_total_tenure_gap_and_recency_adj_avg': 39.441095890410956,
        'profile_relevant_years_of_experience_wgted': 39.441095890410956,  # 0.1356164383561644
        'profile_experience_seniority_progression_rate': 0.000001,
        'profile_position_end_dt_canonical_most_recent': '2018-08-01',
    }

    sample_irrelevant = {
        'ar_candidate_id': 102805174,
        'ar_job_id': 1381,
        'ar_job_title': 'software engineer',
        'ar_job_org_id': 3549,
        'category': 'engineering',
        'job_company_name_canonical': 'playstation',
        'job_company_industry_derived': 'entertainment',
        'job_company_industry_category_derived': 'recreational',
        'job_company_size_derived': '5001-10000',
        'job_company_position_cluster_id': 42,
        'job_company_specialty_cluster_id': 389,
        'criteria': '{"advanced_degree":true,"companies":[{"name":"Google","search_option":"preferred"},{"name":"Airbnb","search_option":"preferred"},{"name":"Amazon","search_option":"preferred"},{"name":"Facebook","search_option":"preferred"},{"name":"Netflix","search_option":"preferred"},{"name":"Microsoft","search_option":"preferred"},{"name":"Lyft","search_option":"preferred"},{"name":"Uber","search_option":"preferred"}],"company_sizes":null,"countries":null,"diversity":false,"fields_of_study":null,"four_year_degree":true,"industries":null,"job_titles":[{"title":"software engineer","search_option":"preferred"}],"locations":[{"location":"San Francisco, California","radius":50}],"management_experience":false,"not_job_hopper":false,"only_current_company_size":false,"schools":null,"skills":[{"name":"Java","search_option":"required"},{"name":"NoSQL","search_option":"required"},{"name":"distributed systems","search_option":"preferred"},{"name":"scalability","search_option":"preferred"},{"name":"cassandra","search_option":"preferred"}],"years_of_relevant_experience_max":13,"years_of_relevant_experience_min":6,"years_of_relevant_experience":1,"years_of_total_experience":0,"additional_notes":"","years_of_experience_max":15,"years_of_experience_min":6}'.lower(),
        'profile_skill_desc': 'supervisory skills | pals | leadership development | medicine | outlook | community outreach | microsoft word | behavioral health | customer service | emr | clinical research | ehr | managed care | healthcare consulting | teaching | physician relations | treatment | training | team building | inpatient | coaching | public health | healthcare information technology | policy | medical billing | healthcare management | case managment | nonprofits | nutrition | hipaa | fundraising | human resources | cpr certified | medicare | nursing | hospitals | medical terminology | budgets | organizational development | public speaking | bls | microsoft office | mental health | executive coaching | non-profits | strategic planning | software documentation | healthcare | powerpoint | grant writing'.lower(),
        'profile_position_company_name_canonical': 'kansas city va medical center',
        'profile_company_size_derived': '10001+',
        'profile_company_industry_derived': 'government administration',
        'profile_company_industry_category_derived': 'public',
        'profile_company_position_cluster_id': None,
        'profile_company_specialty_cluster_id': None,
        'profile_position_most_recent_role': 'registered nurse',
        'profile_experience_most_recent_has_internal_move_6m_ind': 0,
        'profile_fast_riser_wgted': 1,
        'profile_working_for_unicorn_company_wgted': 1,
        'profile_is_entrepreneur_wgted': 0,
        'profile_has_entrepreneurial_experience_wgted': 0,
        'profile_in_high_demand_wgted': 1,
        'profile_is_recent_vip_wgted': 1,
        'profile_position_title_canonical_seniority_level_ord_most_recent_avgflr': 2.0,
        'profile_experience_total_tenure_gap_and_recency_adj_avg': 39.441095890410956,
        'profile_relevant_years_of_experience_wgted': 39.441095890410956,  # 0.1356164383561644
        'profile_experience_seniority_progression_rate': 0.000001,
        'profile_position_end_dt_canonical_most_recent': '2014-06-01',
    }

    sample_default = {
        'ar_candidate_id': 102805174,
        'ar_job_id': 1381,
        'ar_job_title': '',
        'ar_job_org_id': 3549,
        'category': '',
        'job_company_name_canonical': 'abc',
        'job_company_industry_derived': None,
        'job_company_industry_category_derived': None,
        'job_company_size_derived': None,
        'job_company_position_cluster_id': None,
        'job_company_specialty_cluster_id': None,
        'criteria': '{"advanced_degree":true,"companies":[],"company_sizes":null,"countries":null,"diversity":false,"fields_of_study":null,"four_year_degree":true,"industries":null,"job_titles":[],"locations":[{"location":"San Francisco, California","radius":50}],"management_experience":false,"not_job_hopper":false,"only_current_company_size":false,"schools":null,"skills":[],"years_of_relevant_experience_max":13,"years_of_relevant_experience_min":6,"years_of_relevant_experience":1,"years_of_total_experience":0,"additional_notes":"","years_of_experience_max":15,"years_of_experience_min":6}'.lower(),
        'profile_skill_desc': None,
        'profile_position_company_name_canonical': 'def',
        'profile_company_size_derived': None,
        'profile_company_industry_derived': None,
        'profile_company_industry_category_derived': None,
        'profile_company_position_cluster_id': None,
        'profile_company_specialty_cluster_id': None,
        'profile_position_most_recent_role': 'registered nurse',
        'profile_experience_most_recent_has_internal_move_6m_ind': 0,
        'profile_fast_riser_wgted': 1,
        'profile_working_for_unicorn_company_wgted': 1,
        'profile_is_entrepreneur_wgted': 0,
        'profile_has_entrepreneurial_experience_wgted': 0,
        'profile_in_high_demand_wgted': 1,
        'profile_is_recent_vip_wgted': 1,
        'profile_position_title_canonical_seniority_level_ord_most_recent_avgflr': 2.0,
        'profile_experience_total_tenure_gap_and_recency_adj_avg': 39.441095890410956,
        'profile_relevant_years_of_experience_wgted': 39.441095890410956,  # 0.1356164383561644
        'profile_experience_seniority_progression_rate': 0.000001,
        'profile_position_end_dt_canonical_most_recent': '2014-06-01',
    }

    samples = []

    for sample in [sample_good_match, sample_irrelevant, sample_default]:
        sample['job_title_canonical'] = c.canonicalize_title(sample['ar_job_title'])[0]
        sample['job_job_family'] = c.canonicalize_title(sample["job_title_canonical"])[2]
        sample['job_seniority_score'] = c.canonicalize_title(sample["job_title_canonical"])[1]
        sample['job_role'] = c.get_title_role(sample['ar_job_title'])
        sample['job_required_skills'] = c.get_entities_from_criteria(sample["criteria"], "skills", "name", "required")
        sample['job_preferred_skills'] = c.get_entities_from_criteria(sample["criteria"], "skills", "name", "preferred")
        sample['job_all_skills'] = c.concat_string_lists(sample["job_required_skills"], sample["job_preferred_skills"])
        sample['job_all_skills_with_related'] = c.get_similar_terms(sample["job_all_skills"])
        sample['job_required_companies'] = c.get_entities_from_criteria(sample["criteria"], "companies", "name", "required")
        sample['job_preferred_companies'] = c.get_entities_from_criteria(sample["criteria"], "companies", "name", "preferred")
        sample['job_all_companies'] = c.concat_string_lists(sample["job_required_companies"], sample["job_preferred_companies"])
        sample['job_required_companies_px'] = c.tag_text_list(sample['job_required_companies'], "C")
        sample['job_required_companies_px'] = c.tag_text_list(sample['job_preferred_companies'], "C")
        sample['job_all_companies_px'] = c.tag_text_list(sample['job_all_companies'], "C")
        sample['job_required_titles'] = c.get_entities_from_criteria(sample["criteria"], "job_titles", "title", "required")
        sample['job_preferred_titles'] = c.get_entities_from_criteria(sample["criteria"], "job_titles", "title", "preferred")
        sample['job_all_titles'] = c.concat_string_lists(sample["job_required_titles"], sample["job_preferred_titles"])
        sample['job_required_titles_px'] = c.tag_text_list(sample['job_required_titles'], "T")
        sample['job_preferred_titles_px'] = c.tag_text_list(sample['job_preferred_titles'], "T")
        sample['job_all_titles_px'] = c.tag_text_list(sample['job_all_titles'], "T")
        sample['job_required_titles'] = c.get_entities_from_criteria(sample["criteria"], "job_titles", "title", "required")
        sample['job_required_roles'] = c.get_title_role_list(sample["job_required_titles"])
        sample['job_preferred_roles'] = c.get_title_role_list(sample["job_preferred_titles"])
        sample['job_all_roles'] = c.concat_string_lists(sample["job_required_roles"], sample["job_preferred_roles"])
        sample['job_required_roles_px'] = c.tag_text_list(sample['job_required_roles'], "R")
        sample['job_preferred_roles_px'] = c.tag_text_list(sample['job_preferred_roles'], "R")
        sample['job_all_roles_px'] = c.tag_text_list(sample['job_all_roles'], "R")
        sample["job_locations"] = c.get_entities_from_criteria(sample["criteria"], "locations", "location", "")
        sample["job_years_of_experience_min"] = c.get_attribute_from_criteria(sample["criteria"], "years_of_experience_min")
        sample["job_years_of_experience_max"] = c.get_attribute_from_criteria(sample["criteria"], "years_of_experience_max")
        sample["job_years_of_relevant_experience_min"] = c.get_attribute_from_criteria(sample["criteria"], "years_of_relevant_experience_min")
        sample["job_years_of_relevant_experience_max"] = c.get_attribute_from_criteria(sample["criteria"], "years_of_relevant_experience_max")
        sample["job_years_of_relevant_experience"] = c.get_attribute_from_criteria(sample["criteria"], "years_of_relevant_experience")
        sample["profile_job_family"] = c.canonicalize_title(sample["profile_position_most_recent_role"])[2]
        sample["profile_skill_list"] = c.str_to_list(sample["profile_skill_desc"])
        sample["profile_extended_skills"] = c.get_similar_terms(sample["profile_skill_list"])
        sample["profile_matched_skill_set_pct"] = c.check_skill_match(sample["job_required_skills"], sample["job_preferred_skills"], sample["profile_extended_skills"])
        sample["profile_title_match_pct"] = c.check_match(sample["job_all_titles"], sample["profile_position_most_recent_role"])
        sample["profile_most_recent_position_end_days"] = (datetime.now() - datetime.strptime(sample["profile_position_end_dt_canonical_most_recent"], '%Y-%m-%d')).total_seconds() / 86400.0

        if sample['job_company_industry_derived'] and sample["job_company_industry_derived"] == sample["profile_company_industry_derived"]:
            sample["profile_job_industry_match"] = 1
        else:
            sample["profile_job_industry_match"] = 0

        if sample['job_company_industry_category_derived'] and sample["job_company_industry_category_derived"] == sample["profile_company_industry_category_derived"]:
            sample["profile_job_industry_category_match"] = 1
        else:
            sample["profile_job_industry_category_match"] = 0

        if sample['job_company_size_derived'] and sample["job_company_size_derived"] == sample["profile_company_size_derived"]:
            sample["profile_job_company_size_match"] = 1
        else:
            sample["profile_job_company_size_match"] = 0

        if sample['job_company_position_cluster_id'] and sample['job_company_position_cluster_id'] == sample["profile_company_position_cluster_id"]:
            sample["profile_job_company_position_cluster_match"] = 1
        else:
            sample["profile_job_company_position_cluster_match"] = 0

        if sample['job_company_specialty_cluster_id'] and sample['job_company_specialty_cluster_id'] == sample['profile_company_specialty_cluster_id']:
            sample['profile_job_company_position_specialty_match'] = 1
        else:
            sample['profile_job_company_position_specialty_match'] = 0

        if (sample['job_job_family'] and sample['job_job_family'] == sample['profile_job_family']) or (sample['job_role'] and sample['job_role'] == sample['profile_position_most_recent_role']):
            sample['profile_job_job_family_match'] = 1
        else:
            sample['profile_job_job_family_match'] = 0

        sample['relevant_yoe_diff_min'] = sample["profile_relevant_years_of_experience_wgted"] - sample["job_years_of_relevant_experience_min"]
        sample['relevant_yoe_diff_max'] = sample["profile_relevant_years_of_experience_wgted"] - sample["job_years_of_relevant_experience_max"]
        sample['total_yoe_diff_min'] = sample["profile_experience_total_tenure_gap_and_recency_adj_avg"] - sample["job_years_of_experience_min"]
        sample['total_yoe_diff_max'] = sample["profile_experience_total_tenure_gap_and_recency_adj_avg"] - sample["job_years_of_experience_max"]
        sample["total_yoe_diff_mean"] = (sample["total_yoe_diff_min"] + sample["total_yoe_diff_max"]) / 2
        sample["relevant_yoe_diff_mean"] = (sample["relevant_yoe_diff_min"] + sample["relevant_yoe_diff_max"]) / 2

        sample["profile_position_most_recent_titles_px"] = c.tag_text_list(sample["profile_position_most_recent_role"].split(" \| "), "T")
        sample["profile_position_most_recent_roles_px"] = c.tag_text_list(sample["profile_position_most_recent_role"].split(" \| "), "R")

        sample["profile_position_company_name_canonical_px"] = "C_" + sample["profile_position_company_name_canonical"]
        sample["profile_TRCS"] = c.combine(sample["profile_position_most_recent_titles_px"], sample["profile_position_most_recent_roles_px"], sample["profile_position_company_name_canonical_px"], sample["profile_skill_list"])
        sample["profile_TR"] = c.combine(sample["profile_position_most_recent_titles_px"], sample["profile_position_most_recent_roles_px"], None, None)
        sample["profile_C"] = c.combine(None, None, sample["profile_position_company_name_canonical_px"], None)
        sample["profile_S"] = c.combine(None, None, None, sample["profile_skill_list"])

        sample["job_S"] = c.combine(None, None, None, sample["job_all_skills"])
        sample["job_criteria_TRCS"] = c.combine(sample["job_all_titles_px"], sample["job_all_roles_px"], sample["job_all_companies_px"], sample["job_all_skills"])
        sample["job_criteria_TR"] = c.combine(sample["job_all_titles_px"], sample["job_all_roles_px"], None, None)
        sample["job_criteria_C"] = c.combine(None, None, sample["job_all_companies_px"], None)
        sample["job_criteria_C_vector"] = c.get_vectors(sample["job_criteria_C"], sample["job_company_industry_derived"], sample["job_company_size_derived"], "")
        sample["job_criteria_TRCS_vector"] = c.get_vectors(sample["job_criteria_TRCS"], "", "", "")
        sample["job_criteria_TR_vector"] = c.get_vectors(sample["job_criteria_TR"], "", "", "")
        sample["job_S_vector"] = c.get_vectors(sample["job_S"], "", "", "")
        sample["profile_TRCS_vector"] = c.get_vectors(sample["profile_TRCS"], "", "", "")
        sample["profile_TR_vector"] = c.get_vectors(sample["profile_TR"], "", "", sample["job_job_family"])
        sample["profile_C_vector"] = c.get_vectors(sample["profile_C"], sample["profile_company_industry_derived"], sample["profile_company_size_derived"], "")
        sample["profile_S_vector"] = c.get_vectors(sample["profile_S"], "", "", "")

        sample["profile_job_TR_criteria_cosine_similarity"] = c.vector_cosine_similarity(sample["profile_TR_vector"], sample["job_criteria_TR_vector"])
        sample["profile_job_C_criteria_cosine_similarity"] = c.vector_cosine_similarity(sample["profile_C_vector"], sample["job_criteria_C_vector"])
        sample["profile_job_S_cosine_similarity"] = c.vector_cosine_similarity(sample["profile_S_vector"], sample["job_S_vector"])
        sample["profile_all_qualifications_match_score"] = c.check_qualifications_match(sample)

        for col_n in c.NORMALIZE_COLS:
            sample[col_n + '_normed'] = sample[col_n] / c.NORMALIZE_DICT[col_n]

        samples.append(sample)

        # for k in c.FEATURE_ORDER:
        #     print("{} = {}".format(k, sample[k]))
        #     print("")

    preds = c.predict(samples)

    for p in preds:
        print(p)
        print("")


if __name__ == '__main__':
    main()
