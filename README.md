<b>Reference for Basic Algorithms https://github.com/facebookresearch/detectron2</b>


The final classification results obtained are shown in the following figure.

<img width="200" height="400" alt="IMG_20230719_191057_instances_with_polygon_ranked_instances_area_boxes_emptyholes" src="https://github.com/user-attachments/assets/2676eeb7-e5e8-4af8-8248-d0526ce32552" />

Among them, the blue circles represent seedlings without sprouts, the light blue squares represent first-grade seedlings, and the red squares represent third-grade seedlings.


<b>Results under background interference</b>

<img width="200" height="400" alt="IMG_20230903_120233_instances_with_polygon_ranked_instances_area_boxes_emptyholes" src="https://github.com/user-attachments/assets/8aa7f789-b92c-4ca0-91a8-4f4ac4f6f242" >
<img width="200" height="400" alt="IMG_20230903_120233_instances_with_polygon_tray_holes_presence" src="https://github.com/user-attachments/assets/f70ba4cb-6012-49bc-aec1-7c77b0346432" />

The GNN code is the Python file with "GNN" in the title of step10 in the repository https://github.com/Yiyu-Jiang/Seedling-grading. Some visualized grading results can be found in the link https://github.com/Yiyu-Jiang/Seedling-grading/tree/main/mydata/output/step10_graph_merge_area_rank.
