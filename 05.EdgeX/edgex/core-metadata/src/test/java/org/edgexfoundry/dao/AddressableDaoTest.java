/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: support-logging
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.dao;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.mockito.Mockito.when;

import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.test.category.RequiresNone;
import org.edgexfoundry.test.data.AddressableData;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

@Category(RequiresNone.class)
public class AddressableDaoTest {

  private static final String TEST_ID = "123";

  @InjectMocks
  private AddressableDao dao;

  @Mock
  private AddressableRepository repos;

  @Mock
  private DeviceRepository deviceRepos;

  @Mock
  private DeviceServiceRepository deviceServiceRepos;

  private Addressable addressable;

  @Before
  public void setup() throws Exception {
    MockitoAnnotations.initMocks(this);
    addressable = AddressableData.newTestInstance();
  }

  @Test
  public void testGetByIdOrNameWithNull() {
    assertNull("Returned addressable is not null", dao.getByIdOrName(null));
  }

  @Test
  public void testGetByIdOrName() {
    when(repos.findOne(TEST_ID)).thenReturn(addressable);
    addressable.setId(TEST_ID);
    assertEquals("Returned addressable is not expected", addressable,
        dao.getByIdOrName(addressable));
  }

  @Test
  public void testGetByIdOrNameWithNoId() {
    when(repos.findByName(AddressableData.TEST_ADDR_NAME)).thenReturn(addressable);
    assertEquals("Returned addressable is not expected", addressable,
        dao.getByIdOrName(addressable));
  }

  @Test
  public void testIsAddressableAssociatedToDevice() {
    dao.isAddressableAssociatedToDevice(addressable);
  }

  @Test
  public void testIsAddressableAssociatedToDeviceService() {
    dao.isAddressableAssociatedToDeviceService(addressable);
  }

  @Test
  public void testGetOwningServices() {
    dao.getOwningServices(addressable);
  }

  @Test
  public void testGetById() {
    dao.getById(TEST_ID);
  }

  @Test
  public void testGetByName() {
    dao.getByName(AddressableData.TEST_ADDR_NAME);
  }

}
