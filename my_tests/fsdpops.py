import copy
from typing import Tuple, Union

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase

from torchao.dtypes.nf4tensor import _INNER_TENSOR_NAMES_FOR_SHARDING, to_nf4


class TestFSDPOps(TestCase):
    # @parameterized.expand([
    #     (512 * 512,),
    #     ((512 * 512,),),
    #     ((512, 512,),),
    # ])
    def test_torch_chunk_valid(self, input_size: Union[Tuple[int], int]):
        num_chunks = 2
        nf4_tensor = to_nf4(torch.randn(input_size))
        breakpoint()
        chunks = list(torch.chunk(nf4_tensor, num_chunks))
        self.assertEqual(len(chunks), num_chunks)
        if isinstance(input_size, int):
            expected_size0 = input_size // num_chunks
        else:
            expected_size0 = input_size[0] // num_chunks
        for chunk in chunks:
            self.assertEqual(chunk.size(0), expected_size0)

    @parameterized.expand([
        (511 * 512,),
        ((511 * 512,),),
        ((511, 512,),),
    ])
    def test_torch_chunk_invalid_divide(self, input_size: Union[Tuple[int], int]):
        num_chunks = 2
        with self.assertRaisesRegex(
            AssertionError, "Number of scalers must be divisible by scaler block size"
        ):
            nf4_tensor = to_nf4(torch.randn(input_size))
            torch.chunk(nf4_tensor, num_chunks)

    @parameterized.expand([
        ((512, 512, 512),),
    ])
    def test_torch_chunk_invalid_3d(self, input_size: Union[Tuple[int], int]):
        num_chunks = 2
        with self.assertRaisesRegex(AssertionError, "expect input tensor dim <= 2"):
            nf4_tensor = to_nf4(torch.randn(input_size))
            torch.chunk(nf4_tensor, num_chunks)

    def test_tensor_new_zeros_valid(self, input_size: Union[Tuple[int], int]):
        nf4_tensor = to_nf4(torch.randn(input_size))
        breakpoint()
        size = nf4_tensor.size()
        nf4_tensor_zeros = nf4_tensor.new_zeros(input_size)
        # for attr in _INNER_TENSOR_NAMES_FOR_SHARDING:
        #     inner_tensor = getattr(nf4_tensor_zeros, attr)
        #     self.assertEqual(torch.count_nonzero(inner_tensor), 0)
        # expected_size = input_size if not isinstance(input_size, int) else (input_size,)
        # self.assertEqual(nf4_tensor_zeros.size(), torch.Size(expected_size))

    @parameterized.expand([
        (512 * 512,),
        ((512 * 512,),),
        ((512, 512,),),
    ])
    def test_tensor_new_zeros_invalid(self, input_size: Union[Tuple[int], int]):
        if isinstance(input_size, int):
            new_size = input_size + 1
        elif len(input_size) == 1:
            new_size = (input_size[0] + 1,)
        else:
            new_size = (input_size[0] + 1, input_size[1])
        nf4_tensor = to_nf4(torch.randn(input_size))
        with self.assertRaisesRegex(
            NotImplementedError, "aten.new_zeros\\(NF4Tensor\\) with new size"
        ):
            _ = nf4_tensor.new_zeros(new_size)

    def test_tensor_slice_valid(self, input_size: Union[Tuple[int], int]):
        nf4_tensor = to_nf4(torch.randn(input_size))
        orig_attrs, _ = nf4_tensor.__tensor_flatten__()
        orig_sizes = dict(
            [(attr, getattr(nf4_tensor, attr).size()) for attr in orig_attrs]
        )
        end_idx = input_size if isinstance(input_size, int) else input_size[0]

        breakpoint()
        sliced_tensor = nf4_tensor[:end_idx]

        self.assertEqual(nf4_tensor.size(), sliced_tensor.size())
        attrs, _ = sliced_tensor.__tensor_flatten__()
        for attr in attrs:
            orig_storage = getattr(nf4_tensor, attr).untyped_storage().data_ptr()
            sliced_tensor_inner = getattr(sliced_tensor, attr)
            self.assertEqual(
                sliced_tensor_inner.untyped_storage().data_ptr(), orig_storage
            )
            self.assertEqual(sliced_tensor_inner.size(), orig_sizes[attr])

    def test_tensor_slice_1d_invalid(self):
        nf4_tensor = to_nf4(torch.randn(512 * 512))
        with self.assertRaisesRegex(
            NotImplementedError, "aten.slice\\(NF4Tensor\\) with customized step"
        ):
            nf4_tensor[..., ::2]
        with self.assertRaisesRegex(
            NotImplementedError, "aten.slice\\(NF4Tensor\\) with start"
        ):
            nf4_tensor[1:]
        with self.assertRaisesRegex(
            NotImplementedError, "aten.slice\\(NF4Tensor\\) with end"
        ):
            nf4_tensor[:2]

    def test_tensor_slice_2d_invalid(self):
        nf4_tensor = to_nf4(torch.randn((512, 512)))
        with self.assertRaisesRegex(
            NotImplementedError, "aten.slice\\(NF4Tensor\\) with dim"
        ):
            nf4_tensor[:, :511]
        with self.assertRaisesRegex(
            NotImplementedError, "aten.slice\\(NF4Tensor\\) with start"
        ):
            nf4_tensor[1:]
        with self.assertRaisesRegex(
            NotImplementedError, "aten.slice\\(NF4Tensor\\) with end"
        ):
            nf4_tensor[:2]


    def test_tensor_view_valid(self, input_size: Union[Tuple[int], int]):
        nf4_tensor = to_nf4(torch.randn(input_size))
        viewed_tensor = nf4_tensor.view(-1)
        self.assertEqual(viewed_tensor.dim(), 1)
        self.assertEqual(viewed_tensor.numel(), math.prod(input_size))
        for attr in _INNER_TENSOR_NAMES_FOR_SHARDING:
            inner_tensor = getattr(viewed_tensor, attr)
            self.assertEqual(inner_tensor.size(0), inner_tensor.numel())


    def test_tensor_view_invalid(self, input_size: Union[Tuple[int], int]):
        nf4_tensor = to_nf4(torch.randn(input_size))
        if len(input_size) == 1:
            with self.assertRaisesRegex(
                NotImplementedError, "aten.view\\(NF4Tensor\\) with size"
            ):
                nf4_tensor.view(input_size)
        if len(input_size) == 2:
            with self.assertRaisesRegex(
                NotImplementedError, "aten.view\\(NF4Tensor\\) with len\\(size\\)"
            ):
                breakpoint()
                nf4_tensor.view(input_size)

    @parameterized.expand([
        (512 * 512,),
        ((512 * 512,),),
        ((512, 512,),),
    ])
    def test_tensor_as_strided_valid(self, input_size: Union[Tuple[int], int]):
        nf4_tensor = to_nf4(torch.randn(input_size))
        nf4_tensor_strided = torch.as_strided(
            nf4_tensor,
            nf4_tensor.size(),
            nf4_tensor.stride(),
            nf4_tensor.storage_offset(),
        )
        self.assertEqual(nf4_tensor_strided.size(), nf4_tensor.size())
        self.assertEqual(nf4_tensor_strided.stride(), nf4_tensor.stride())
        self.assertEqual(
            nf4_tensor_strided.storage_offset(), nf4_tensor.storage_offset()
        )
        for attr in _INNER_TENSOR_NAMES_FOR_SHARDING:
            inner_tensor_orig = getattr(nf4_tensor, attr)
            inner_tensor_strided = getattr(nf4_tensor_strided, attr)
            self.assertEqual(inner_tensor_strided.size(), inner_tensor_orig.size())
            self.assertEqual(inner_tensor_strided.stride(), inner_tensor_orig.stride())
            self.assertEqual(
                inner_tensor_strided.storage_offset(),
                inner_tensor_orig.storage_offset(),
            )

    @parameterized.expand([
        (512 * 512,),
        ((512 * 512,),),
        ((512, 512,),),
    ])
    def test_tensor_as_strided_invalid(self, input_size: Union[Tuple[int], int]):
        nf4_tensor = to_nf4(torch.randn(input_size))
        if len(input_size) == 1:
            size = (input_size[0] - 1,)
        else:
            size = (input_size[0] - 1, input_size[1])
        with self.assertRaisesRegex(
            NotImplementedError, "aten.as_strided\\(NF4Tensor\\) different numel"
        ):
            torch.as_strided(
                nf4_tensor, size, nf4_tensor.stride(), nf4_tensor.storage_offset()
            )
        with self.assertRaisesRegex(
            NotImplementedError,
            "aten.as_strided\\(NF4Tensor\\) only support original storage offset",
        ):
            torch.as_strided(nf4_tensor, nf4_tensor.size(), nf4_tensor.stride(), 1)

        if len(input_size) == 2:
            with self.assertRaisesRegex(
                NotImplementedError,
                "aten.as_strided\\(NF4Tensor\\) only support continuous stride",
            ):
                stride = (nf4_tensor.stride()[1], nf4_tensor.stride()[0])
                torch.as_strided(
                    nf4_tensor, nf4_tensor.size(), stride, nf4_tensor.storage_offset()
                )

    def test_pin_memory(self):
        nf4_tensor = to_nf4(torch.randn(512 * 512))
        self.assertFalse(nf4_tensor.is_pinned())

        nf4_tensor = nf4_tensor.pin_memory()
        self.assertTrue(nf4_tensor.is_pinned())

        nf4_tensor = to_nf4(torch.randn(512 * 512, device="cuda"))
        self.assertFalse(nf4_tensor.is_pinned())

    def test_to_cuda(self):
        nf4_tensor = to_nf4(torch.randn(512 * 512))
      #  self.assertEqual(nf4_tensor.device.type, "cpu")
        breakpoint()
        nf4_tensor = nf4_tensor.to("cuda", non_blocking=True)
        self.assertEqual(nf4_tensor.device.type, "cuda")
        #self.assertEqual(type(nf4_tensor), NF4Tensor)
        nf4_tensor.get_original_weight()  # make sure we can dequantize

        # nf4_tensor = to_nf4(torch.randn(512 * 512))
        # self.assertEqual(nf4_tensor.device.type, "cpu")
        # nf4_tensor = nf4_tensor.to("cuda")
        # self.assertEqual(nf4_tensor.device.type, "cuda")
        # self.assertEqual(type(nf4_tensor), NF4Tensor)
        # nf4_tensor.get_original_weight()

        # nf4_tensor = to_nf4(torch.randn(512 * 512))
        # self.assertEqual(nf4_tensor.device.type, "cpu")
        # nf4_tensor = nf4_tensor.to("cuda", torch.bfloat16)
        # self.assertEqual(nf4_tensor.device.type, "cuda")
        # self.assertEqual(nf4_tensor.dtype, torch.bfloat16)
        # self.assertEqual(type(nf4_tensor), torch.Tensor)  # dequantized

    def test_to_cpu(self):
        nf4_tensor = to_nf4(torch.randn(512 * 512, device="cuda"))
        nf4_tensor = nf4_tensor.cpu()
        self.assertEqual(nf4_tensor.device.type, "cpu")
        for attr in _INNER_TENSOR_NAMES_FOR_SHARDING:
            inner_tensor = getattr(nf4_tensor, attr)
            self.assertEqual(inner_tensor.device.type, "cpu")
        nf4_tensor.get_original_weight()  # make sure we can dequantize

    def test_to_module(self):
        linear = nn.Linear(512, 512, bias=False)
        linear.weight = nn.Parameter(
            to_nf4(linear.weight.detach()), requires_grad=False
        )
        linear.cuda()
        self.assertEqual(linear.weight.device.type, "cuda")
        weight = linear.weight.get_original_weight()
        self.assertEqual(weight.device.type, "cuda")

        linear.cpu()
        self.assertEqual(linear.weight.device.type, "cpu")
        weight = linear.weight.get_original_weight()
        self.assertEqual(weight.device.type, "cpu")

        linear = nn.Linear(512, 512, bias=False)
        linear.weight = nn.Parameter(
            to_nf4(linear.weight.detach()), requires_grad=False
        )
        linear.to("cuda")
        self.assertEqual(linear.weight.device.type, "cuda")
        weight = linear.weight.get_original_weight()
        self.assertEqual(weight.device.type, "cuda")

        linear.to("cpu")
        self.assertEqual(linear.weight.device.type, "cpu")
        weight = linear.weight.get_original_weight()
        self.assertEqual(weight.device.type, "cpu")

    def test_tensor_deepcopy(self, input_size: Union[Tuple[int], int]):
        nf4_orig = to_nf4(torch.randn(input_size, device="cuda"))
        nf4_clone = copy.deepcopy(nf4_orig)
        self.assertEqual(
            nf4_clone.get_original_weight(), nf4_orig.get_original_weight()
        )

if __name__ == "__main__":
    tests = TestFSDPOps()
    #tests.test_torch_chunk_valid((512, 512))
    #tests.test_tensor_new_zeros_valid((512, 512))
    #tests.test_tensor_slice_valid((512, 512))
    #tests.test_tensor_view_invalid((512, 512))
    tests.test_to_cpu()