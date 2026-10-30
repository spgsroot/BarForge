"""Crop geometry, neighboring censorship and recoverable incremental dataset publication."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from barforge.automation import CropOptions, PageAnalysis, prepare_page, export_dataset
from barforge.autocensor import CensorOptions, Detection, bars_from_detections
from barforge.imaging import render_pair
from barforge.models import Bar, Document, Rect
from barforge.storage import load_session, save_session


class AutomationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "neutral.png"
        Image.new("RGB", (1600,1000), (80,80,80)).save(self.source)

    def tearDown(self):
        self.temp.cleanup()

    def page(self, detections, size=(1600,1000)):
        return PageAnalysis(str(self.source),size,tuple(detections),("fixture",),1,0.)

    def test_edge_crops_keep_entire_anchor_and_stay_inside_small_source(self):
        for rect in (Rect(0,0,80,150),Rect(300,160,100,140),Rect(0,0,400,300)):
            with self.subTest(rect=rect):
                page = self.page([Detection("genital","penis",.9,rect)],(400,300))
                candidates = prepare_page(page,CensorOptions(),CropOptions(aspect_ratio=.75),"edit {color}")
                self.assertEqual(len(candidates),1)
                crop = candidates[0].document.crop
                self.assertLessEqual(crop.x,rect.x)
                self.assertLessEqual(crop.y,rect.y)
                self.assertGreaterEqual(crop.x+crop.width,rect.x+rect.width)
                self.assertGreaterEqual(crop.y+crop.height,rect.y+rect.height)
                self.assertGreaterEqual(crop.x,0)
                self.assertGreaterEqual(crop.y,0)
                self.assertLessEqual(crop.x+crop.width,400)
                self.assertLessEqual(crop.y+crop.height,300)

    def test_neighbor_regions_have_same_censorship_in_crop_as_in_full_page(self):
        detections = [Detection("genital","penis",.9,Rect(500,400,100,100)),
                      Detection("genital","pussy",.8,Rect(750,400,100,100))]
        options = CensorOptions(color="#000000",angle=65)
        page = self.page(detections)
        proposals = prepare_page(page,options,CropOptions(context=6),"edit {color}")
        first = min(proposals,key=lambda candidate:candidate.document.crop.x)
        self.assertEqual(first.detection_indices,(0,1))
        bars = bars_from_detections(detections,*page.size,options)
        _,full = render_pair(Document(str(self.source),bars))
        clean,censored = render_pair(first.document)
        crop = first.document.crop
        np.testing.assert_array_equal(np.asarray(censored),np.asarray(full.crop((crop.x,crop.y,crop.x+crop.width,crop.y+crop.height))))
        mask = np.any(np.asarray(clean) != np.asarray(censored),axis=2)
        neighbor_left = 750-crop.x
        self.assertTrue(np.any(mask[:,neighbor_left:]))

    def test_overlapping_crops_are_deduplicated_and_cap_is_confidence_ranked(self):
        detections = [Detection("genital","penis",score,Rect(x,300,30,50))
                      for x,score in ((100,.6),(105,.5),(500,.9),(900,.8),(1300,.7))]
        result = prepare_page(self.page(detections),CensorOptions(),CropOptions(min_side=128,max_crops=2),"edit")
        self.assertEqual(len(result),2)
        self.assertEqual({item.score for item in result},{.9,.8})
        duplicate = prepare_page(self.page(detections[:2]),CensorOptions(),CropOptions(min_side=128),"edit")
        self.assertEqual(len(duplicate),1)
        self.assertEqual(duplicate[0].detection_indices,(0,1))

    def test_empty_and_filtered_pages_do_not_invent_pairs(self):
        for detections in ([],[Detection("nipple","nipple_f",.9,Rect(100,100,50,50))],
                           [Detection("genital","penis",.1,Rect(100,100,50,50))]):
            self.assertEqual(prepare_page(self.page(detections),CensorOptions(),CropOptions(mode="page"),"edit"),[])

    def test_whole_page_option_is_unique_and_included_in_the_cap(self):
        page = self.page([Detection("genital","penis",.9,Rect(200,300,100,100))])
        result = prepare_page(page,CensorOptions(),CropOptions(include_page=True,max_crops=1),"edit {color}")
        self.assertEqual(len(result),1)
        self.assertEqual(result[0].document.crop,Rect(0,0,1600,1000))
        self.assertEqual(result[0].document.caption,"edit white")

    def test_crop_controls_reject_unusable_geometry(self):
        for options in (CropOptions(context=float("nan")),CropOptions(context=.5),CropOptions(min_side=0),
                        CropOptions(max_crops=True),CropOptions(mode="invent"),CropOptions(aspect_ratio=-1)):
            with self.subTest(options=options),self.assertRaises(ValueError):
                prepare_page(self.page([]),CensorOptions(),options,"edit")

    def test_natural_aspect_crop_follows_local_anatomy_not_the_scene_box(self):
        organ = Detection("genital","penis",.9,Rect(500,400,300,60))
        scene = Detection("act","make_love",.99,Rect(0,0,1600,1000))
        crops = CropOptions(context=1.35,min_side=32,aspect_ratio=0)
        proposals = prepare_page(self.page([scene,organ]),CensorOptions(),crops,"edit")
        self.assertEqual(len(proposals),1)
        crop = proposals[0].document.crop
        self.assertLessEqual(crop.width,405)
        self.assertLessEqual(crop.height,81)
        self.assertGreaterEqual(crop.x+crop.width,800)
        self.assertGreaterEqual(crop.y+crop.height,460)
        self.assertEqual(prepare_page(self.page([scene]),CensorOptions(),crops,"edit"),[])

    def documents(self):
        return [Document(str(self.source),[Bar(100,100,100,5,"#000000",30)],Rect(50,50,400,300),"first"),
                Document(str(self.source),[Bar(700,200,180,7,"#ffffff",65)],Rect(600,100,450,400),"second")]

    def test_export_keeps_source_and_prior_runs_and_uses_private_resumable_documents(self):
        documents = self.documents()
        source_bytes = self.source.read_bytes()
        output = self.root / "dataset"
        first = export_dataset(documents,output)
        saved_first = {path.relative_to(output).as_posix():path.read_bytes() for path in output.rglob("*") if path.is_file()}
        second = export_dataset(documents,output)
        self.assertEqual([doc.export_name for doc in first.documents],["001","002"])
        self.assertEqual([doc.export_name for doc in second.documents],["003","004"])
        self.assertNotEqual(first.session_path,second.session_path)
        for relative,data in saved_first.items():
            self.assertEqual((output/relative).read_bytes(),data)
        self.assertEqual(self.source.read_bytes(),source_bytes)
        self.assertTrue(all(doc.export_name is None for doc in documents))
        restored = load_session(Path(first.session_path))
        self.assertEqual([doc.signature() for doc in restored.documents],[doc.signature() for doc in first.documents])
        self.assertTrue(all(doc.is_exported for doc in restored.documents))
        for document in restored.documents:
            clean,censored = render_pair(document)
            with Image.open(output/"clean"/f"{document.export_name}.png") as actual:
                np.testing.assert_array_equal(np.asarray(actual),np.asarray(clean))
            with Image.open(output/"censored"/f"{document.export_name}.png") as actual:
                np.testing.assert_array_equal(np.asarray(actual),np.asarray(censored))

    def test_cancellation_retains_the_completed_pair_and_its_checkpoint(self):
        stopped = [False]
        def progress(done,total):
            if done == 1:
                stopped[0] = True
        result = export_dataset(self.documents(),self.root/"dataset",progress=progress,cancelled=lambda:stopped[0])
        self.assertTrue(result.cancelled)
        self.assertEqual([doc.export_name for doc in result.documents],["001"])
        self.assertEqual([doc.export_name for doc in load_session(Path(result.session_path)).documents],["001"])
        self.assertEqual({path.name for path in (self.root/"dataset"/"censored").glob("*.png")},{"001.png"})

    def test_one_unreadable_source_does_not_erase_or_block_other_pairs(self):
        documents = self.documents()
        documents.insert(1,Document(str(self.root/"missing.png"),caption="missing"))
        result = export_dataset(documents,self.root/"dataset")
        self.assertEqual(len(result.errors),1)
        self.assertEqual(result.errors[0][0],str(self.root/"missing.png"))
        self.assertEqual([doc.caption for doc in result.documents],["first","second"])
        self.assertEqual([doc.caption for doc in load_session(Path(result.session_path)).documents],["first","second"])

    def test_failed_checkpoint_reports_the_already_saved_pair_and_stops(self):
        calls = [0]
        def fail_second_save(session,path):
            calls[0] += 1
            if calls[0] == 2:
                raise OSError("checkpoint unavailable")
            save_session(session,path)
        with patch("barforge.automation.save_session",side_effect=fail_second_save):
            result = export_dataset(self.documents(),self.root/"dataset")
        self.assertEqual(len(result.errors),1)
        self.assertEqual([doc.export_name for doc in result.documents],["001"])
        self.assertTrue((self.root/"dataset"/"clean"/"001.png").is_file())
        self.assertTrue((self.root/"dataset"/"censored"/"001.png").is_file())
        self.assertFalse((self.root/"dataset"/"clean"/"002.png").exists())


if __name__ == "__main__":
    unittest.main()
