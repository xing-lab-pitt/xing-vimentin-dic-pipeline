dir_path='/home/dap182/cluster/xing/image_analysis/image_data/40x_large_calcein_time_lapse_training_datset/tifs/a549_vim_rfp_2ng_24hr_091721/*C1*';

crop_path='/home/dap182/cluster/xing/image_analysis/image_data/40x_large_calcein_time_lapse_training_datset/crops/a549_vim_rfp_2ng_24hr_091721';
files = dir(dir_path);

for num=1:size(files,1)
    file_name=files(num).name;
    
    file_dir=files(num).folder;
    
    dic=imread(strcat(file_dir,'/',file_name));
        
    imshow(imadjust(dic));
    
   
    crop_list={};
    crop_size=100000;
    crop_num=1;
        while(crop_size>=10000)
            [pick_data, rect] = imcrop();
            
            crop_size=numel(pick_data);
            
            crop_dic=imcrop(dic,rect);
            
            crop_num=crop_num+1;
            
            %imwrite(uint16(crop_dic),strcat(crop_path,file_name,'cr',num2str(crop_size),'.png'));
             
        end
    
end