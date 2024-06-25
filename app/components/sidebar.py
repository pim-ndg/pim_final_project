import streamlit as st

def show():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Introduction", "Buy-Again", "Coupon",
                                           "Customer-Segment", "Buy-Again Predict"])

    if selection == "Buy-Again":
        st.sidebar.write("Navigating to the buy-again recommendations page.")
        import pages.buy_again as buy_again
        buy_again.show()
    elif selection == "Introduction":
        st.sidebar.write("Navigating to the introduction page.")
        import pages.introduction as introduction
        introduction.show()
    elif selection == "Coupon":
        st.sidebar.write("Navigating to the coupon and discount prediction page.")
        import pages.coupon_discount as coupon_discount
        coupon_discount.show()
    elif selection == "Customer-Segment":
        st.sidebar.write("Navigating to the customer segmentation page.")
        import pages.customer_segmentation as cust_seg
        cust_seg.show()
    elif selection == "Buy-Again Predict":
        st.sidebar.write("Navigating to the buy again based on customer segmentation prediction page,")
        import pages.buy_again_cust_seg as ba_seg
        ba_seg.show()