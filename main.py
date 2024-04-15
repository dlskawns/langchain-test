import streamlit as st
from utils import print_messages, StreamHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os

st.set_page_config(page_title="ChatGPT", page_icon="Q")
st.title("ChatGPT")

# API KEY 설정
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# session state

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# 채팅 대화기록을 저장하는 store 세션 상태 변수
if 'store' not in st.session_state:
    st.session_state['store'] = dict()


# 카톡방으로 치면 카톡방의 이름이라고 생각하면 된다. 
with st.sidebar:
    session_id = st.text_input("Session Id", value="abc123")

    clear_btn = st.button('대화기록 초기화')
    if clear_btn:
        st.session_state['message'] = []
        st.experimental_rerun()

print_messages()


def get_session_history(session_ids:str) -> BaseChatMessageHistory:
    print(session_ids)
    if session_ids not in st.session_state['store']: #세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성해 store에 저장
        st.session_state['store'][session_ids] = ChatMessageHistory()
    return st.session_state['store'][session_ids] # 해당 세션 ID에 대한 세션 기록 반환 

if user_input := st.chat_input("메시지를 입력해주세요."):
    st.chat_message('user').write(f"{user_input}")
    # st.session_state['messages'].append(('user',user_input))
    st.session_state['messages'].append(ChatMessage(role='user', content=user_input))

    with st.chat_message('assistant'):
        stream_handler = StreamHandler(st.empty())
        # 1. 모델 생성
        llm = ChatOpenAI(streaming = True, callbacks=[stream_handler])

        # 2. 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    '질문에 짧고 간결하게 답변해 주세요'
                ),
                # 대화 기록을 변수로 사용, history가 MessageHistory의 Key가 됨
                MessagesPlaceholder(variable_name = "history"),
                ("human", "{question}"), # 사용자의 질문을 입력
            ]
        )
        chain = prompt | llm
        # response = chain.invoke({'question': user_input})

        chain_with_memory = RunnableWithMessageHistory(chain,
                                    get_session_history,
                                    input_messages_key='question',
                                    history_messages_key='history',)
        
        response = chain_with_memory.invoke(
        #수학 관련 질문 "코사인의 의미는 무엇인가요?"를 입력으로 전달합니다.
        {"question": user_input},
        # 설정 정보로 세션 ID "abc123"을 전달합니다.
        config={"configurable":{"session_id": session_id}}
        )
        msg = response.content
        # st.write(msg)
        # st.session_state['messages'].append(('assistant',user_input))
        st.session_state['messages'].append(ChatMessage(role='assistant', content=msg))