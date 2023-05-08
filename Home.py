import streamlit as st
st.set_page_config(page_title='바쁜사람들', layout="wide", initial_sidebar_state="collapsed")


STYLE = """
#key {
    color: #000;
    background-color: #FAF3DD;
    text-decoration: none;
}

.title {
    font-size: 24px;
    margin-top: 10px;
    font-weight: medium;
}

.callout {
    padding: 20px;
    background-color: #F8F8F8;
    border-left: 4px solid #3B81F5;
    margin: 1em 0em;
    color: black;
}

.team-member {
    display: flex;
    align-items: center;
    margin-bottom: 20px;
    border-radius: 10px;
    padding: 10px 35px;
    background-color: #fafafa;
    -webkit-backdrop-filter: blur(5px);
}

.member-info {
  text-align: left;
  margin-left: 10px;
  padding: 20px;
}

.team-member .profile-image {
    width: 150px;
    height: 150px;
    border-radius: 50%;
}

.introduction {
  margin-top: 10px;
  font-size: 16px;
}

.cta-container-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    background-image: linear-gradient(rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.2)));
    background-size: cover;
    padding: 10px;
    height: flex;
    border-radius: 10px;
    margin: 10px 0px;
}

.cta-container-wrapper.cta-container1 {
    background-image: url('https://i.pinimg.com/564x/7d/ef/e5/7defe5156cc72de1264011624e73e44c.jpg');
}

.cta-container-wrapper.cta-container2 {
    background-image: url('https://i.pinimg.com/564x/1f/87/d9/1f87d9f026f352bf2b662db576503186.jpg');
}

.cta-container-wrapper.cta-container3 {
    background-image: url('https://i.pinimg.com/564x/15/82/7c/15827c807ea133c3345b951a47a5cb7c.jpg');
}

.rounded-image-container {
  width: 100%;
  height: 0;
  padding-bottom: calc(100% * (3/12));
  position: relative;
  overflow: hidden;
  border-radius: 10px;
  margin : 1em 0em;
}

.rounded-image-container2 {
  width: 100%;
  height: 0;
  padding-bottom: calc(100% * (3/7));
  position: relative;
  overflow: hidden;
  border-radius: 10px;
  margin : 1em 0em;
}

.rounded-image {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 10px;
}

</style>
"""

st.markdown(f"<style>{STYLE}</style>", unsafe_allow_html=True)
st.markdown("""
  <div class="rounded-image-container">
    <img src="https://github.com/hollyisyoon/streamlit/blob/main/%E1%84%8A%E1%85%A5%E1%86%B7%E1%84%82%E1%85%A6%E1%84%8B%E1%85%B5%E1%86%AF%20(1).png?raw=true" class="rounded-image" alt="썸네일">
  </div>
""", unsafe_allow_html=True)

st.markdown(f"""
    <h2>안녕하세요 👋🏻 바쁜 사람들입니다</h2>""",
    unsafe_allow_html=True
)

st.markdown("""
    <div class='callout'> 
    <li>바쁜 사람들은 1인 셀러들이 사업의 규모와 관계없이 <b>데이터에 기반한 의사결정</b>을 내릴 수 있는 환경을 제공하기 위해 제작되었습니다.</li> 
    <li>비즈니스 성장을 위해서는 다양한 비즈니스 데이터를 통해 문제점을 파악하고 알맞은 액션 플랜을 설계하는 것이 중요합니다.</li>
    <li>바쁜 사람들은 유통 채널별 리뷰 데이터는 물론 경쟁사 리뷰 데이터에 대한 분석, 외부 데이터를 통한 트렌드 및 키워드 분석 결과를 <b>분석가 없이도 손쉽게 볼 수 있는 대시보드</b>를 제공합니다.</li>
    </div>""", unsafe_allow_html=True)

####CTA버튼####
cta_container1, cta_container2, cta_container3 = st.beta_columns(3)

# CTA container 1
with cta_container1:
    st.markdown(
        """
        <div class="cta-container-wrapper cta-container1">
            <div style='padding: 10px; border-radius: 10px'>
                <h2 style='color: white; text-align: center'>User Guide</h2>
                <p style='color: white; text-align: center'>바쁜 사람들이 처음이라면?</p>
                <p style='text-align: center'>
                    <a href='https://flowerbloomingtime.notion.site/e79013461ccb4f75bcb051a8b49f14a2' target='_blank'>
                        <button style='background-color: white; color: #3B81F5; padding: 8px 16px; border-radius: 5px; border: none; font-weight: bold; cursor: pointer'>
                            살펴보기
                        </button>
                    </a>
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with cta_container2:
    st.markdown(
        """
        <div class="cta-container-wrapper cta-container2">
            <div style='padding: 10px; border-radius: 10px'>
                <h2 style='color: white; text-align: center'>Case Study</h2>
                <p style='color: white; text-align: center'>비즈니스 시나리오에 적용해보자!</p>
                <p style='text-align: center'>
                    <a href='https://flowerbloomingtime.notion.site/Case-Study-ec47f93c2eb94327afd950cb197fb970' target='_blank'>
                        <button style='background-color: white; color: #3B81F5; padding: 8px 16px; border-radius: 5px; border: none; font-weight: bold; cursor: pointer'>
                            시작하기
                        </button>
                    </a>
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with cta_container3:
    st.markdown(
        """
        <div class="cta-container-wrapper cta-container3">
            <div style='padding: 10px; border-radius: 10px'>
                <h2 style='color: white; text-align: center'>Final PPT</h2>
                <p style='color: white; text-align: center'>기획부터 제작과정이 궁금하다면?</p>
                <p style='text-align: center'>
                    <a href='https://flowerbloomingtime.notion.site/Case-Study-ec47f93c2eb94327afd950cb197fb970' target='_blank'>
                        <button style='background-color: white; color: #3B81F5; padding: 8px 16px; border-radius: 5px; border: none; font-weight: bold; cursor: pointer'>
                            다운받기
                        </button>
                    </a>
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ppt_url = "https://docs.google.com/presentation/d/15cUqztq9yXrbzor0rgyU6jXpH4puvrvY4ZSOcGz6S_s/edit#slide=id.g24035bb5175_0_8"
# st.markdown(f'<iframe src="{ppt_url}" width="800" height="600" frameborder="0" scrolling="no"></iframe>', unsafe_allow_html=True)

####멤버 소개###
st.markdown("---")
st.markdown("""<h2>만든 사람들</h2>""", unsafe_allow_html=True)

profile1, profile2 = st.beta_columns(2)
with profile1:
    st.markdown('''   
    <div class="team-member">
        <img src="https://media.licdn.com/dms/image/D5603AQGLWfWNmVBIYQ/profile-displayphoto-shrink_800_800/0/1665667362702?e=1689206400&v=beta&t=2RzA1JP0qxRbKImCayGJqEMuFZwZqbTR8QYGLAyz5Rg" class="profile-image">
        <div class="member-info">
            <h3> 윤훈영 </h3>
            <p><code>Product</code> <code>SaaS</code> <code>광고</code> <code>BI</code> 에 관심이 많습니다.
            🚀 매 순간, 끊임없이 성장을 즐기는 사람들과 챌린징한 과제가 있는 환경에서 재밌고 도전적으로 일해보고 싶어요!</p>
            <details>
                <summary><b>Feel Free..</b></summary>
                <p>To Reach Me.. 💙적극 구직 중입니다💙 -> <a id="key" href="https://linkedin.com/in/hoonyoungyoon/" target="_blank">Linkedin</a></p>
            </details>
        </div>
    </div>
    ''', unsafe_allow_html=True)

with profile2:
    st.markdown('''   
        <div class="team-member">
            <img src="https://blogpfthumb-phinf.pstatic.net/MjAyMzAzMjRfMTk0/MDAxNjc5NjU5NTY5MTU4.cTkkysdHNu7TxVaSnGM-WaFdYOKz-XlUYZHiqk9Y7Hkg.n_MJqADE_VCqT4AXrxnocLWp0hvS5xIeDieN5DRc2Ewg.PNG.kinghy00/profileImage.png" class="profile-image">
            <div class="member-info">
                <h3> 김희연 </h3>
                <p><li>Goal: 바쁜 서비스 기획자되기</li>
                <li>Dream: 검은머리가 파뿌리가 될때까지 배우고 성장하기</li>
                </p>
            </div>
        </div>
        ''', unsafe_allow_html=True)

profile1, profile2 = st.beta_columns(2)
with profile1:
    st.markdown('''   
    <div class="team-member">
        <img src="https://github.com/seoinhyeok96/mypage/blob/main/d80f4ece871ddb035af1c1dab314bcf8-sticker.png?raw=true" class="profile-image">
        <div class="member-info">
            <h3> 서인혁 </h3>
            <p><li>자기소개: 태블로 마스터가 되고싶은 그로스해커</li>
            <li>꿈: 데이터 분석으로 숨겨진 인사이트 찾기</li>
            <li>목표:  데이터를 활용할 수 있고 보수도 안정인 곳에 취업하기</li></p>
        </div>
    </div>
    ''', unsafe_allow_html=True)

with profile2:
    st.markdown('''   
    <div class="team-member">
        <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FEKyyU%2FbtsereKWyAc%2F923KaucLcqdl0X07gbPKs1%2Fimg.jpg" class="profile-image">
        <div class="member-info">
            <h3> 김효정 </h3>
            <p>데이터 분석 분야에 대한 지식과 경험을 갖춘 전문가를 꿈꾸고 있습니다. 다양한 팀원들과 함께 혹은 따로, 책임감 있게 일합니다.</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)

profile1, profile2 = st.beta_columns(2)
with profile1:
    st.markdown('''   
    <div class="team-member">
        <img src="https://avatars.githubusercontent.com/u/94737255?v=4" class="profile-image">
        <div class="member-info">
            <h3> 박민정 </h3>
            <p>데이터 수집부터 모델링, 시각화까지의 과정을 전문적으로 수행하고 싶은 미래의 데이터 분석가입니다. <code>통계학</code>, <code>머신 러닝</code>, <code>데이터 시각화</code>등 다양한 분야에 관심을 가지고 있습니다.</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)

with profile2:
    st.markdown('''   
    <div class="team-member">
        <img src="https://i.ytimg.com/vi/QgB10EW-KWc/maxresdefault.jpg" class="profile-image">
        <div class="member-info">
            <h3> 송준태 </h3>
            <p>팔 하나로 산을 움직일 수 있는 힘이 있으며 펀치를 맞은 상대는 지평선 끝까지 날아간다고 한다. 생각하는 것보다도 먼저 손이 나온다.... 이거 없다고?</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)

####코멘트###
st.markdown("---")
st.markdown("""
<h2>베타 테스터 모집 중</h2>
  <div class="rounded-image-container2">
    <img src="https://raw.githubusercontent.com/hollyisyoon/streamlit/main/%E1%84%83%E1%85%A2%E1%84%89%E1%85%B5%E1%84%87%E1%85%A9%E1%84%83%E1%85%B3%20(1).png" class="rounded-image" 
    alt="메인 이미지"></img>
  </div>
""", unsafe_allow_html=True)
st.markdown("""<detail><summary>설문폼 제출하기</summary>
<iframe src="https://docs.google.com/forms/d/e/1FAIpQLSfVMkoqtFTniuMFjlRkC1ToQBSmu9esdYtiXQ3-4Lj0hvwrsA/viewform?embedded=true" width="100%" height="1035" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>
</detail>""", unsafe_allow_html=True)
