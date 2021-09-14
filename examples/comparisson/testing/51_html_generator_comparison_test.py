from util.templite import Templite
import os




if __name__ == '__main__':


    template=Templite.from_file("templates/amt-it-place.html")
    amt_configuration = True

    if amt_configuration:
        total_num_question = 11
        base_url_images = "https://amt-it-place-apo.s3.eu-west-2.amazonaws.com/it-place/"
        example_url_images = "https://amt-it-place-apo.s3.eu-west-2.amazonaws.com/examples/"
        comparison_imgs = [["${gif_left_"+str(i)+"}", "${gif_right_"+str(i)+"}"] for i in range(1, total_num_question+1)]
    else:
        total_num_question = 11
        base_url_images = "../../../images/amt_sample/"
        example_url_images = "../../../images/"
        comparison_imgs = [["17DRP5sb8fy-livingroom/laying_bed/place/body_1_opt2.gif", "17DRP5sb8fy-livingroom/laying_bed/it/body_1_opti_smplx.gif"],
                           ["room_0/sitting_stool_one_foot_floor/place/body_0_opt2.gif", "/room_0/sitting_stool_one_foot_floor/it/body_0_opti_smplx.gif"],
                           ["hotel_0/sitting_small_table/place/body_0_opt2.gif","hotel_0/sitting_small_table/it/body_0_opti_smplx.gif"],
                           ["office_2/standup_hand_on_furniture/place/body_1_opt2.gif","office_2/standup_hand_on_furniture/it/body_1_opti_smplx.gif"],
                           ["apartment_1/reaching_out_mid_down/body_fake.gif","apartment_1/reaching_out_mid_down/it/body_1_opti_smplx.gif"],
                           ["MPH1Library/laying_on_sofa/place/body_2_opt2.gif","MPH1Library/laying_on_sofa/it/body_2_opti_smplx.gif"],
                           ["X7HyMhZNoso-livingroom_0_16/reaching_out_mid/body_fake.gif","X7HyMhZNoso-livingroom_0_16/reaching_out_mid/it/body_0_opti_smplx.gif"],
                           ["17DRP5sb8fy-familyroomlounge/laying_on_sofa/place/body_2_opt2.gif","17DRP5sb8fy-familyroomlounge/laying_on_sofa/it/body_2_opti_smplx.gif"],
                           ["N0SittingBooth/sitting_small_table/place/body_2_opt2.gif","N0SittingBooth/sitting_small_table/it/body_2_opti_smplx.gif"],
                           ["X7HyMhZNoso-livingroom_0_16/standing_up/place/body_2_opt2.gif","X7HyMhZNoso-livingroom_0_16/standing_up/it/body_2_opti_smplx.gif"],
                           ["N3OpenArea/sitting_comfortable/place/body_1_opt2.gif","N3OpenArea/sitting_comfortable/it/body_1_opti_smplx.gif"]]






    assert total_num_question == len(comparison_imgs)

    with open('amt-it-place.html', 'w') as the_file:
        the_file.write(template.render(total_num_question = total_num_question,
                                       base_url_images = base_url_images,
                                       comparison_imgs=comparison_imgs,
                                       example_url_images=example_url_images))