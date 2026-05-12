# LIDC malignancy score를 실제 학습 label로 변환하는 기준
# LIDC malignancy score : 폐 결절(nodule)이 악성 종양(폐암)일 가능성을 영상의학 전문의가 평가한 점수


# =========================================================
# LIDC Malignancy Score Label Policy
# =========================================================
#
# LIDC malignancy score:
#
# 1 -> Highly Unlikely for Cancer
# 2 -> Moderately Unlikely
# 3 -> Indeterminate
# 4 -> Moderately Suspicious
# 5 -> Highly Suspicious
#
# =========================================================


# =========================================================
# Method A
# =========================================================
#
# 1,2 -> benign (0)
# 4,5 -> malignant (1)
# 3   -> 제외
#
# 가장 많이 사용되는 baseline 방식
# =========================================================
def score_to_label_method_a(score: int):

    # benign
    if score in [1, 2]:
        return 0

    # malignant
    if score in [4, 5]:
        return 1

    # score 3은 ambiguous 하므로 제외
    return None


# =========================================================
# Method C
# =========================================================
#
# 1 -> benign (0)
# 5 -> malignant (1)
# 2,3,4 -> 제외
#
# 더 엄격한(high confidence) 설정
# =========================================================
def score_to_label_method_c(score: int):

    # benign
    if score == 1:
        return 0

    # malignant
    if score == 5:
        return 1

    # ambiguous 구간 제거
    return None


# =========================================================
# Method 이름으로 함수 가져오기
# =========================================================
def get_label_policy(method: str):

    # Method A
    if method == "a":
        return score_to_label_method_a

    # Method C
    elif method == "c":
        return score_to_label_method_c

    # 지원하지 않는 method
    else:
        raise ValueError(
            f"Unsupported method: {method}"
        )


# =========================================================
# 실행 테스트 → python -m src.preprocessing.label_policy
# =========================================================
if __name__ == "__main__":

    print("===== LABEL POLICY TEST =====")

    print()

    test_scores = [1, 2, 3, 4, 5]

    # -----------------------------------------------------
    # Method A Test
    # -----------------------------------------------------
    print("Method A")

    for score in test_scores:

        label = score_to_label_method_a(score)

        print(
            f"score={score} -> label={label}"
        )

    print()

    # -----------------------------------------------------
    # Method C Test
    # -----------------------------------------------------
    print("Method C")

    for score in test_scores:

        label = score_to_label_method_c(score)

        print(
            f"score={score} -> label={label}"
        )

    print()

    print("Label Policy Test Success")