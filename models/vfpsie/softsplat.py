#!/usr/bin/env python

import collections
import cupy
import os
import re

import torch
import typing

# Provide the legacy compiler entrypoint on CuPy versions that removed it.
try:
    import inspect

    from cupy.cuda import compiler as _cupy_compiler

    if not hasattr(cupy.cuda, "compile_with_cache"):
        def _compile_with_cache(source, options=(), arch=None, cache_dir=None):
            if hasattr(_cupy_compiler, "_compile_module_with_cache"):
                target = _cupy_compiler._compile_module_with_cache
                # Pass only compiler options supported by the installed CuPy version.
                accepted = inspect.signature(target).parameters
                candidate_kwargs = {"options": options, "arch": arch,
                                   "cache_dir": cache_dir, "backend": "nvrtc"}
                kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}
                return target(source, **kwargs)

            return cupy.RawModule(code=source, options=options, backend="nvrtc")

        cupy.cuda.compile_with_cache = _compile_with_cache  # type: ignore[attr-defined]
except Exception:
    # Defer compiler errors until the CUDA path is requested.
    pass


objCudacache = {}


def cuda_int32(intIn:int):
    return cupy.int32(intIn)


def cuda_float32(fltIn:float):
    return cupy.float32(fltIn)


def cuda_kernel(strFunction:str, strKernel:str, objVariables:typing.Dict):
    if 'device' not in objCudacache:
        objCudacache['device'] = torch.cuda.get_device_name()

    strKey = strFunction

    #use string record the function and parameter
    for strVariable in objVariables:
        objValue = objVariables[strVariable]

        strKey += strVariable

        if objValue is None:
            continue

        elif type(objValue) == int:
            strKey += str(objValue)

        elif type(objValue) == float:
            strKey += str(objValue)

        elif type(objValue) == bool:
            strKey += str(objValue)

        elif type(objValue) == str:
            strKey += objValue

        elif type(objValue) == torch.Tensor:
            strKey += str(objValue.dtype)
            strKey += str(objValue.shape)
            strKey += str(objValue.stride())

        elif True:
            print(strVariable, type(objValue))
            assert(False)


    strKey += objCudacache['device']

    if strKey not in objCudacache:
        for strVariable in objVariables:
            objValue = objVariables[strVariable]

            if objValue is None:
                continue

            elif type(objValue) == int:
                strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))

            elif type(objValue) == float:
                strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))

            elif type(objValue) == bool:
                strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))

            elif type(objValue) == str:
                strKernel = strKernel.replace('{{' + strVariable + '}}', objValue)

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.uint8:
                strKernel = strKernel.replace('{{type}}', 'unsigned char')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float16:
                strKernel = strKernel.replace('{{type}}', 'half')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float32:
                strKernel = strKernel.replace('{{type}}', 'float')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float64:
                strKernel = strKernel.replace('{{type}}', 'double')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.int32:
                strKernel = strKernel.replace('{{type}}', 'int')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.int64:
                strKernel = strKernel.replace('{{type}}', 'long')

            elif type(objValue) == torch.Tensor:
                print(strVariable, objValue.dtype)
                assert(False)

            elif True:
                print(strVariable, type(objValue))
                assert(False)


        while True:
            objMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

            if objMatch is None:
                break

            intArg = int(objMatch.group(2))

            strTensor = objMatch.group(4)
            intSizes = objVariables[strTensor].size()
            strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg] if torch.is_tensor(intSizes[intArg]) == False else intSizes[intArg].item()))

        while True:
            objMatch = re.search('(OFFSET_)([0-4])(\()', strKernel)

            if objMatch is None:
                break

            intStart = objMatch.span()[1]
            intStop = objMatch.span()[1]
            intParentheses = 1

            while True:
                intParentheses += 1 if strKernel[intStop] == '(' else 0
                intParentheses -= 1 if strKernel[intStop] == ')' else 0

                if intParentheses == 0:
                    break

                intStop += 1

            intArgs = int(objMatch.group(2))
            strArgs = strKernel[intStart:intStop].split(',')

            assert(intArgs == len(strArgs) - 1)

            strTensor = strArgs[0]
            intStrides = objVariables[strTensor].stride()

            strIndex = []

            for intArg in range(intArgs):
                strIndex.append('((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg] if torch.is_tensor(intStrides[intArg]) == False else intStrides[intArg].item()) + ')')

            strKernel = strKernel.replace('OFFSET_' + str(intArgs) + '(' + strKernel[intStart:intStop] + ')', '(' + str.join('+', strIndex) + ')')

        while True:
            objMatch = re.search('(VALUE_)([0-4])(\()', strKernel)

            if objMatch is None:
                break

            intStart = objMatch.span()[1]
            intStop = objMatch.span()[1]
            intParentheses = 1

            while True:
                intParentheses += 1 if strKernel[intStop] == '(' else 0
                intParentheses -= 1 if strKernel[intStop] == ')' else 0

                if intParentheses == 0:
                    break

                intStop += 1

            intArgs = int(objMatch.group(2))
            strArgs = strKernel[intStart:intStop].split(',')

            assert(intArgs == len(strArgs) - 1)

            strTensor = strArgs[0]
            intStrides = objVariables[strTensor].stride()

            strIndex = []

            for intArg in range(intArgs):
                strIndex.append('((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg] if torch.is_tensor(intStrides[intArg]) == False else intStrides[intArg].item()) + ')')

            strKernel = strKernel.replace('VALUE_' + str(intArgs) + '(' + strKernel[intStart:intStop] + ')', strTensor + '[' + str.join('+', strIndex) + ']')

        objCudacache[strKey] = {
            'strFunction': strFunction,
            'strKernel': strKernel
        }

    return strKey


@cupy.memoize(for_each_device=True)
def cuda_launch(strKey:str):
    if 'CUDA_HOME' not in os.environ:
        os.environ['CUDA_HOME'] = cupy.cuda.get_cuda_path()

    return cupy.cuda.compile_with_cache(objCudacache[strKey]['strKernel'], tuple(['-I' + os.environ['CUDA_HOME'], '-I' + os.environ['CUDA_HOME'] + '/include'])).get_function(objCudacache[strKey]['strFunction'])


def softsplat(tenIn:torch.Tensor, tenFlow:torch.Tensor, tenMetric:torch.Tensor, strMode:str):
    assert(strMode.split('-')[0] in ['sum', 'avg', 'linear', 'soft'])

    if strMode == 'sum': assert(tenMetric is None)
    if strMode == 'avg': assert(tenMetric is None)
    if strMode.split('-')[0] == 'linear': assert(tenMetric is not None)
    if strMode.split('-')[0] == 'soft': assert(tenMetric is not None)

    if strMode == 'avg':
        tenIn = torch.cat([tenIn, tenIn.new_ones([tenIn.shape[0], 1, tenIn.shape[2], tenIn.shape[3]])], 1)

    elif strMode.split('-')[0] == 'linear':
        tenIn = torch.cat([tenIn * tenMetric, tenMetric], 1)

    elif strMode.split('-')[0] == 'soft':
        tenIn = torch.cat([tenIn * tenMetric.exp(), tenMetric.exp()], 1)


    tenOut,Mask = softsplat_func.apply(tenIn, tenFlow)

    if strMode.split('-')[0] in ['avg', 'linear', 'soft']:
        tenNormalize = tenOut[:, -1:, :, :]

        if len(strMode.split('-')) == 1:
            tenNormalize = tenNormalize + 0.0000001

        elif strMode.split('-')[1] == 'addeps':
            tenNormalize = tenNormalize + 0.0000001

        elif strMode.split('-')[1] == 'zeroeps':
            tenNormalize[tenNormalize == 0.0] = 1.0

        elif strMode.split('-')[1] == 'clipeps':
            tenNormalize = tenNormalize.clip(0.0000001, None)


        tenOut = tenOut[:, :-1, :, :] / tenNormalize

    return tenOut,Mask


class softsplat_func(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, tenIn, tenFlow):
        tenOut = tenIn.new_zeros([tenIn.shape[0], tenIn.shape[1], tenIn.shape[2], tenIn.shape[3]])
        Mask=tenIn.new_zeros([tenIn.shape[0], tenIn.shape[1], tenIn.shape[2], tenIn.shape[3]])
        tenTag=tenIn.new_ones([tenIn.shape[0], tenIn.shape[1], tenIn.shape[2], tenIn.shape[3]])+1


        if tenIn.is_cuda == True:
            deterministic = os.environ.get("EVCOM_DETERMINISTIC_SOFTSPLAT") == "1"
            launch_grid = (1, 1, 1) if deterministic else (
                int((tenOut.nelement() + 512 - 1) / 512), 1, 1
            )
            deterministic_block = int(
                os.environ.get("EVCOM_DETERMINISTIC_SOFTSPLAT_BLOCK", "32")
            )
            if deterministic and deterministic_block not in (32, 64, 128, 256, 512):
                raise ValueError(
                    "EVCOM_DETERMINISTIC_SOFTSPLAT_BLOCK must be a power of two "
                    "between 32 and 512"
                )
            launch_block = (deterministic_block, 1, 1) if deterministic else (512, 1, 1)
            cuda_launch(cuda_kernel('softsplat_out', '''
                extern "C" __global__ void __launch_bounds__(512) softsplat_out(
                    const int n,
                    const {{type}}* __restrict__ tenIn,
                    const {{type}}* __restrict__ tenFlow,
                    {{type}}* __restrict__ tenOut,
                    const {{type}}* __restrict__ tenTag,
                    {{type}}* __restrict__ Mask
                ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
                    const int intN = ( intIndex / SIZE_3(tenOut) / SIZE_2(tenOut) / SIZE_1(tenOut) ) % SIZE_0(tenOut);
                    const int intC = ( intIndex / SIZE_3(tenOut) / SIZE_2(tenOut)                  ) % SIZE_1(tenOut);
                    const int intY = ( intIndex / SIZE_3(tenOut)                                   ) % SIZE_2(tenOut);
                    const int intX = ( intIndex                                                    ) % SIZE_3(tenOut);

                    assert(SIZE_1(tenFlow) == 2);

                    {{type}} fltX = ({{type}}) (intX) + VALUE_4(tenFlow, intN, 0, intY, intX);
                    {{type}} fltY = ({{type}}) (intY) + VALUE_4(tenFlow, intN, 1, intY, intX);

                    if (isfinite(fltX) == false) { return; }
                    if (isfinite(fltY) == false) { return; }

                    {{type}} fltIn = VALUE_4(tenIn, intN, intC, intY, intX);

                    {{type}} fltTag = VALUE_4(tenTag, intN, intC, intY, intX);

                    int intNorthwestX = (int) (floor(fltX));
                    int intNorthwestY = (int) (floor(fltY));
                    int intNortheastX = intNorthwestX + 1;
                    int intNortheastY = intNorthwestY;
                    int intSouthwestX = intNorthwestX;
                    int intSouthwestY = intNorthwestY + 1;
                    int intSoutheastX = intNorthwestX + 1;
                    int intSoutheastY = intNorthwestY + 1;

                    {{type}} fltNorthwest = (({{type}}) (intSoutheastX) - fltX) * (({{type}}) (intSoutheastY) - fltY);
                    {{type}} fltNortheast = (fltX - ({{type}}) (intSouthwestX)) * (({{type}}) (intSouthwestY) - fltY);
                    {{type}} fltSouthwest = (({{type}}) (intNortheastX) - fltX) * (fltY - ({{type}}) (intNortheastY));
                    {{type}} fltSoutheast = (fltX - ({{type}}) (intNorthwestX)) * (fltY - ({{type}}) (intNorthwestY));

                    if ((intNorthwestX >= 0) && (intNorthwestX < SIZE_3(tenOut)) && (intNorthwestY >= 0) && (intNorthwestY < SIZE_2(tenOut))) {
                        atomicAdd(&tenOut[OFFSET_4(tenOut, intN, intC, intNorthwestY, intNorthwestX)], fltIn * fltNorthwest);
                    }

                    if ((intNortheastX >= 0) && (intNortheastX < SIZE_3(tenOut)) && (intNortheastY >= 0) && (intNortheastY < SIZE_2(tenOut))) {
                        atomicAdd(&tenOut[OFFSET_4(tenOut, intN, intC, intNortheastY, intNortheastX)], fltIn * fltNortheast);
                    }

                    if ((intSouthwestX >= 0) && (intSouthwestX < SIZE_3(tenOut)) && (intSouthwestY >= 0) && (intSouthwestY < SIZE_2(tenOut))) {
                        atomicAdd(&tenOut[OFFSET_4(tenOut, intN, intC, intSouthwestY, intSouthwestX)], fltIn * fltSouthwest);
                    }

                    if ((intSoutheastX >= 0) && (intSoutheastX < SIZE_3(tenOut)) && (intSoutheastY >= 0) && (intSoutheastY < SIZE_2(tenOut))) {
                        atomicAdd(&tenOut[OFFSET_4(tenOut, intN, intC, intSoutheastY, intSoutheastX)], fltIn * fltSoutheast);
                    }




                     if ((intNorthwestX >= 0) && (intNorthwestX < SIZE_3(Mask)) && (intNorthwestY >= 0) && (intNorthwestY < SIZE_2(Mask))) {
                        atomicAdd(&Mask[OFFSET_4(Mask, intN, intC, intNorthwestY, intNorthwestX)], fltTag * fltNorthwest);
                    }

                    if ((intNortheastX >= 0) && (intNortheastX < SIZE_3(Mask)) && (intNortheastY >= 0) && (intNortheastY < SIZE_2(Mask))) {
                        atomicAdd(&Mask[OFFSET_4(Mask, intN, intC, intNortheastY, intNortheastX)], fltTag * fltNortheast);
                    }

                    if ((intSouthwestX >= 0) && (intSouthwestX < SIZE_3(Mask)) && (intSouthwestY >= 0) && (intSouthwestY < SIZE_2(Mask))) {
                        atomicAdd(&Mask[OFFSET_4(Mask, intN, intC, intSouthwestY, intSouthwestX)], fltTag * fltSouthwest);
                    }

                    if ((intSoutheastX >= 0) && (intSoutheastX < SIZE_3(Mask)) && (intSoutheastY >= 0) && (intSoutheastY < SIZE_2(Mask))) {
                        atomicAdd(&Mask[OFFSET_4(Mask, intN, intC, intSoutheastY, intSoutheastX)], fltTag * fltSoutheast);
                    }


                } }
            ''', {
                'tenIn': tenIn,
                'tenFlow': tenFlow,
                'tenOut': tenOut,
                'tenTag':tenTag,
                'Mask':Mask,
            }))(
                # Preserve the selected CUDA accumulation order.
                grid=launch_grid,
                block=launch_block,
                args=[cuda_int32(tenOut.nelement()), tenIn.data_ptr(), tenFlow.data_ptr(), tenOut.data_ptr(),tenTag.data_ptr(),Mask.data_ptr()],
                stream=collections.namedtuple('Stream', 'ptr')(torch.cuda.current_stream().cuda_stream)
            )

        elif tenIn.is_cuda != True:
            assert(False)



        Mask=(Mask.mean(1)==0)

        return tenOut,Mask
