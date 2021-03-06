__kernel void calc_pi(int n_element,
                      __local float *local_pi,
                      __global float *pi_4)
{

   /* Get the work group size, work-item ID and work-group ID */
   int wg_size = get_local_size(0);
   int wi_id = get_local_id(0);
   int wg_id = get_group_id(0);

   /* Initialize the local portion of the Pi calculation */
   local_pi[wi_id] = 0.0;

   /*for (int i = 0; i < wg_size; i++)
   {
      local_pi[i] = 0;
   }*/

   /* Wait for all local data to be cleared before continuing. */
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Compute the fractional components of Pi for this partial sum */
   int glob_id = (int)get_global_id(0);
   float s;

   s = 1.0 / (4.0 * (2.0 * glob_id) + 1.0) - 1.0 / (4.0 * (2.0 * glob_id) + 3.0) +
       1.0 / (4.0 * (2.0 * glob_id + 1.0) + 1.0) - 1.0 / (4.0 * (2.0 * glob_id + 1.0) + 3.0);
   local_pi[wi_id] = s;

   /* Make sure local processing has completed */
   barrier(CLK_GLOBAL_MEM_FENCE);

   /* Perform local reduction. Sum the local_pi values into local_pi[0] */
   if (wi_id == 0){
       for(int i = 1; i < wg_size; i++){
           local_pi[0] += local_pi[i];
       }

       pi_4[wg_id] = local_pi[0];
   }
}
