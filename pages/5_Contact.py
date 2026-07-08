import os
import streamlit as st
import config

# Configure page
st.set_page_config(
    page_title="Contact & Developer Details - OncoVision",
    page_icon="https://cdn-icons-png.flaticon.com/512/2877/2877840.png",
    layout="wide"
)

# Load CSS
css_path = os.path.join(config.ASSETS_DIR, "css", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="padding-bottom: 20px;">
        <h1 style="font-size: 2.5rem; margin-bottom: 5px;">Contact & Developer Information</h1>
        <p style="color: #94A3B8; font-size: 1.1rem; margin-top: 0;">
            Reach out for research collaborations, portal customization, or developer inquiries.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="color: #FF006E; margin-top:0;">Developer Profile</h3>
            <p style="color: #E2E8F0; font-size: 0.95rem; line-height: 1.6;">
                <b>Lead AI Engineer & System Architect</b><br/>
                Specializing in Computer Vision, Deep Learning, and Medical AI solutions.
            </p>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
                Interested in taking this project to production or integrating it with institutional clinical systems? 
                Feel free to connect or review the open-source repository details below.
            </p>
            <hr style="border-color: rgba(255,255,255,0.05); margin: 15px 0;"/>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.95rem; color: #E2E8F0;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 10px 0; font-weight: bold; color: #FF006E; width: 30%;">Email</td>
                    <td style="padding: 10px 0;"><a href="mailto:developer@oncovision.ai" style="color: #3A86FF; text-decoration: none;">developer@oncovision.ai</a></td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 10px 0; font-weight: bold; color: #FF006E;">GitHub</td>
                    <td style="padding: 10px 0;"><a href="https://github.com" target="_blank" style="color: #3A86FF; text-decoration: none;">github.com/oncovision-ai</a></td>
                </tr>
                <tr>
                    <td style="padding: 10px 0; font-weight: bold; color: #FF006E;">LinkedIn</td>
                    <td style="padding: 10px 0;"><a href="https://linkedin.com" target="_blank" style="color: #3A86FF; text-decoration: none;">linkedin.com/in/oncovision</a></td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="color: #3A86FF; margin-top:0;">Quick Inquiry Form</h3>
            <p style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 15px;">
                Submit your feedback or research queries directly to our engineering team.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Simple form
    with st.form("contact_form"):
        name = st.text_input("Full Name", placeholder="Jane Doe")
        email = st.text_input("Email Address", placeholder="jane@example.com")
        subject = st.selectbox("Inquiry Subject", ["Research Collaboration", "Bug Report", "Custom Model Integration", "General Question"])
        message = st.text_area("Detailed Message", placeholder="Write your message here...")
        
        submit_btn = st.form_submit_button("Submit Message")
        
        if submit_btn:
            if not name or not email or not message:
                st.error("Please fill in all fields before submitting.")
            else:
                st.success("Thank you! Your message has been logged. We will reach out shortly.")
                st.toast("Message submitted!")
