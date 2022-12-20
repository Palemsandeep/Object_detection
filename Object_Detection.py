import streamlit as st
from PIL import Image
import glob
import matplotlib.pyplot as plt
import cv2
import imutils
import pytesseract
from configparser import ConfigParser
import os


def main():
	st.set_page_config(layout="wide")
	global select2
	global select3
	global select4
	global folder_path
	global slect5
	global all_imgs
	global dict_file
	output_file = open("output.txt",'w')
	all_imgs = []
	folder_path = None
	select2 = None 
	select3 = None
	select4 = None
	select5 = None
	dict_file = {}

	with open('style.css') as f:
		st.markdown (f"<style>{f.read()}</style>",unsafe_allow_html=True)

	def Reset_fun():
		pass
		st.session_state['key1']="Select the problem Statement"
		st.session_state['key2']="Home"
		st.session_state['key3']="Library Used"
		st.session_state['key4']="Model Implemented"
		st.session_state['key5']="GCP"



	col1, col2, col3 = st.columns([3,8,2])
	with col1:
		st.write("")
	with col2:
		img = Image.open("Deepsphere_image.png")
		st.image(img,width=900)
	with col3:
		st.write("")

	st.markdown("<h1 style='text-align: center; color: Black;font-size: 29px;font-family:IBM Plex Sans;'>Learn to Build Industry Standard Data Science Applications</h1>", unsafe_allow_html=True)
	st.markdown("<p style='text-align: center; color: Blue;font-size: 29px;font-family:IBM Plex Sans;'>MLOPS Built On Google Cloud and Streamlit</p>", unsafe_allow_html=True)
	st.markdown("<p style='text-align: center; color: black; font-size:22px;font-family:IBM Plex Sans;'><span style='font-weight: bold'>Problem Statement: </span>Develop a Machine Learning Application for vehicle number plate Classfication</p>", unsafe_allow_html=True)
	st.markdown("______________________________________________________________________________________________________________________________________________")
	
	c11,c12,c13,c14,c15 = st.columns((3,5,1,1,1))
	with c11:
		st.write("")
		st.write("")
		st.write("")
		st.markdown("##### **Problem Statement**")
	with c12:
		select1 = st.selectbox("",['Select the problem Statement','classify the number plate'],key = "key1")
	with c13:
		st.markdown("")
	with c14:
		st.markdown("")
	with c15:
		st.markdown("")

	st_list1 = ['classify the number plate']
	
	c21,c22,c23,c24,c25 = st.columns(((3,5,1,1,1)))
	with c21:
		if select1 in st_list1:
			st.write("")
			st.write("")
			st.write("")
			st.markdown("##### **Problem type**")
	with c22:
		if select1 in st_list1:
			select2 = st.selectbox("",['Select the problem type','Classfication',])
	with c23:
		st.markdown("")
	with c24:
		st.markdown("")
	with c25:
		st.markdown("")


	st_list2 = ['Classfication']
	c31, c32, c33 ,c34,c35= st.columns(((3,5,1,1,1)))
	with c31:
		if select2 in st_list2:
			st.write("")
			st.write("")
			st.write("")
			st.markdown("##### **Model Selection**")
	with c32:
		if select2 in st_list2:
			select3 = st.selectbox("",['Select the Model','Tesseract-Ocr','Easy-Ocr'])
	with c33:
		st.markdown("")
	with c34:
		st.markdown("")
	with c35:
		st.markdown("")

	st_list3 = ['Tesseract-Ocr','East-Ocr']
	c41,c42,c43,c44,c45 = st.columns(((3,5,1,1,1)))
	with c41:
		if select3 in st_list3:
			st.write("")
			st.write("")
			st.write("")
			st.markdown("##### **Upload File**")
	with c42:
		image_file_type = []
		if select3 in st_list3:
			folder_path = st.text_input("Enter the Path")
	with c43:
		st.markdown("")
	with c44:
		if select3 in st_list3:
			st.write("")
			st.write("")
			st.write("")
			select4 = st.button('Preview')
	with c45:
		st.markdown("")
	if select4 is True:
		cd1,cd2,cd3,cd4,cd5 = st.columns((2,2,2,2,2))
		images=  folder_path +"/*.*" 
		all_imgs = glob.glob(images ,recursive=True)
		# print("The total images present in the dataset: {}".format(len(all_imgs)))
		Display_Images= all_imgs[0:5]
		for i in range(len(Display_Images)):
			with cd1:

				st.image(Image.open(Display_Images[i]))
			with cd2:
				st.image(Image.open(Display_Images[i+1]))
			with cd3:
				st.image(Image.open(Display_Images[i+2]))
			with cd4:
				st.image(Image.open(Display_Images[i+3]))
			with cd5:
				st.image(Image.open(Display_Images[i+4]))
				break
	c51,c52,c53,c54,c55 = st.columns(((3,5,1,1,1)))
	with c51:
		if folder_path is not None:
			st.write("")
			st.write("")
			st.write("")
			st.markdown("##### **Feature Engineering**")
	with c52:
		if folder_path is not None:
			st.multiselect('Image Features',["Licence Number",'State'])
	with c53:
		st.markdown("")
	with c54:
		st.markdown("")
	with c55:
		st.markdown("")

	c51,c52,c53,c54,c55 = st.columns(((3,5,1,1,1)))
	with c51:
		if folder_path is not None:
			st.write("")
			st.write("")
			st.write("")
			st.markdown("##### **Hyper Parameter Tunning**")
	with c52:
		if folder_path is not None:
			st.selectbox('HperParameters',["HyperParameter1","HyperParameter2"])
	with c53:
		st.markdown("")
	with c54:
		st.markdown("")
	with c55:
		st.markdown("")

	c61,c62,c63,c64,c65 = st.columns(((3,3,10,3,3)))
	with c61:
		if folder_path is not None:
			st.write("")
			st.write("")
			st.write("")
			st.markdown("##### **Model Engineering**")
	with c62:
		st.markdown("")
	with c63:
		if folder_path is not None:
			st.write("")
			st.write("")
			st.write("")
			select5 = st.button("Execute the Model")
			if select5 != None:
				st.write(select5)
	with c64:
		st.markdown("")
	with c65:
		st.markdown("")
	if select5 is True:
		images=  folder_path +"/*.*" 
		all_imgs = glob.glob(images ,recursive=True)

		state_dictionary = {'AN': 'Andaman and Nicobar Islands', 
		                    'AP': 'Andhra Pradesh', 
		                    'AR': 'Arunachal Pradesh', 
		                    'AS': 'Assam',
		                    'BR': 'Bihar',
		                    'CH': 'Chandigarh', 
		                    'CT': 'Chhattisgarh', 
		                    'DN': 'Dadra and Nagar Haveli',
		                    'DD': 'Daman and Diu', 
		                    'DL': 'Delhi', 
		                    'GA': 'Goa',
		                    'GJ': 'Gujarat', 
		                    'HR': 'Haryana',
		                    'HP': 'Himachal Pradesh',
		                    'JK': 'Jammu and Kashmir', 
		                    'JH': 'Jharkhand', 
		                    'KA': 'Karnataka',
		                    'KL': 'Kerala', 
		                    'LD': 'Lakshadweep',
		                    'MP': 'Madhya Pradesh ', 
		                    'MH': 'Maharashtra',
		                    'MN': 'Manipur', 
		                    'ML': 'Meghalaya', 
		                    'MZ': 'Mizoram', 
		                    'NL': 'Nagaland', 
		                    'OR': 'Odisha', 
		                    'PY': 'Puducherry', 
		                    'PB': 'Punjab', 
		                    'RJ': 'Rajasthan',
		                    'SK': 'Sikkim', 
		                    'TN': 'Tamil Nadu', 
		                    'TG': 'Telangana', 
		                    'TR': 'Tripura', 
		                    'UP': 'Uttar Pradesh', 
		                    'UT': 'Uttarakhand', 
		                    'WB': 'West Bengal'}

		
		for i in range(len(all_imgs)):
			try:
				#Taking in our image input and resizing its width to 300 pixels
				image = cv2.imread(all_imgs[i])
				image = imutils.resize(image, width=300 )

				#Converting the input image to greyscale

				gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


				#Reducing the noise in the greyscale image

				gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17) 


				#Detecting the edges of the smoothened image

				edged = cv2.Canny(gray_image, 30, 200) 


				#Finding the contours from the edged image

				cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
				image1=gray_image.copy()
				cv2.drawContours(image1,cnts,-1,(0,255,0),3)


				#Sorting the identified contours

				cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:30]
				screenCnt = 0
				image2 = gray_image.copy()
				cv2.drawContours(image2,cnts,-1,(0,255,0),3)


				#Finding the contour with four sides and Cropping the rectangular part identified as license plate

				i=7
				for c in cnts:
				      perimeter = cv2.arcLength(c, True)
				      approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
				      if len(approx) == 4: 
				              screenCnt = approx
				      x,y,w,h = cv2.boundingRect(c)
				      new_img=gray_image[y:y+h,x:x+w]
				      cv2.imwrite('./'+str(i)+'.png',new_img)
				      i+=1
				      break

				#Drawing the selected contour on the original image

				cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)


				#Extracting text from the image of the cropped license plate usinig tessaract

				Cropped_loc = './7.png'
				plate = pytesseract.image_to_string(Cropped_loc,lang ='eng',config ='--oem 3 --psm 6')
				# Removing special Characters from the text

				bad_chars = [';', ':', '!', "*","”","“",'»']
				text = plate

				# using filter() to  remove bad_chars
				text = ''.join((filter(lambda i: i not in bad_chars,text)))
				text = text.split()
				first_in_list = text[0]
				state = state_dictionary[first_in_list[:2]]
				text = " ".join(line.strip() for line in plate.splitlines())
				dict_file[text] = state
				output_file.write(f"\n Number_plate : {text}  state:  {state}") 
			except Exception as e:
				continue
		st.write(dict_file.keys())
	output_file.close()
	
	c61,c62,c63 = st.columns((7,3,5))
	with c61:
		st.markdown("")
	with c62:
		if folder_path is not None:
			st.markdown("")
			st.markdown("")
			output_file = open('output.txt','r')
			select6 = st.download_button("Download",output_file,file_name="OutPut.txt",mime='text')
	with c63:
		st.markdown("")
	output_file.close()

	st.sidebar.selectbox('Menu',["Home",'Model Validation','Download Model Outcome','Data Visualization','Deploy the Model'],key='key2')
	st.sidebar.selectbox("",['Library Used','Streamlit','Pandas','Ipython.display','sklearn.linear_model'],key='key3')
	st.sidebar.selectbox("",['Model Implemented','Decision Tree','Random Forest','Logistic Regression'],key='key4')
	st.sidebar.selectbox("",['GCP','VM Instance','Computer Engine','Cloud Storage'],key='key5')
	c51,c52,c53 = st.sidebar.columns((1,1,1))
	with c51:
		pass
	with c52:
		st.sidebar.button("clear/Reset",on_click=Reset_fun)
	with c53:
		pass



if __name__ == '__main__':
	main()