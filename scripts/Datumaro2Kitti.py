import json
import os
import sys


def json_to_kitti(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    labels = data['categories']['label']['labels']
    os.makedirs(output_dir, exist_ok=True)

    total = 0
    empty = 0
    ann_total = 0

    for item in data['items']:
        item_id = item['id']
        annotations = item.get('annotations', [])

        output_path = os.path.join(output_dir, f"{item_id}.txt")
        total += 1

        with open(output_path, 'w') as f_out:
            if not annotations:
                empty += 1
                continue

            for annotation in annotations:
                label_id = annotation['label_id']
                label_name = labels[label_id]['name']

                position = annotation['position']
                rotation = annotation['rotation']
                scale = annotation['scale']

                truncated = 0
                occluded = 1 if annotation['attributes']['occluded'] else 0
                alpha = rotation[2]
                bbox_left = bbox_top = bbox_right = bbox_bottom = 0.0
                height = scale[2]
                width = scale[0]
                length = scale[1]
                x, y, z = position
                rotation_y = rotation[2]

                f_out.write(f"{label_name} {truncated} {occluded} {alpha} "
                            f"{bbox_left} {bbox_top} {bbox_right} {bbox_bottom} "
                            f"{height} {width} {length} {x} {y} {z} {rotation_y}\n")
                ann_total += 1

    print(f"done. total={total} labeled={total - empty} empty={empty} boxes={ann_total}")


if __name__ == '__main__':
    json_path = sys.argv[1] if len(sys.argv) > 1 else 'default.json'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'kitti_labels'
    json_to_kitti(json_path, output_dir)
